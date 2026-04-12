# SPDX-License-Identifier: Apache-2.0
"""
Collective KV Cache Reuse for TokenDance (Component 2).

Instead of processing each agent's shared blocks independently (N separate
RoPE rotations + important-position selections), the KV Collector groups
the N requests in an All-Gather round and performs these operations once.

Key classes:

* :class:`ReusePlan` — metadata produced by collective reuse that bridges
  into Diff-Aware Storage (Component 3).
* :class:`KVCollector` — groups compatible requests, drives layerwise
  collective retrieval + compute in lockstep.
* :func:`find_compatible_groups` — partitions a batch of requests into
  groups eligible for collective reuse.

Reference: TokenDance paper, Section 4.2 & Section 5 —
Collective KV Cache Reuse.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Data structures
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class AgentRequest:
    """Lightweight descriptor for one agent sub-request in a round.

    Attributes:
        req_id: Unique request identifier.
        round_id: Identifier for the All-Gather round this request belongs to.
        prompt_tokens: Full token sequence (with ``<TTSEP>`` separators).
        prompt_len: Number of tokens in the prompt.
        cached_span: Number of leading tokens already cached.
        slot_mapping: Slot IDs in the paged KV cache (list of ints).
        segment_hashes: Content hashes of the segments in this prompt
            (as produced by :class:`RoundAwareSegmentDatabase`).
    """

    req_id: str
    round_id: str
    prompt_tokens: List[int]
    prompt_len: int
    cached_span: int
    slot_mapping: List[int]
    segment_hashes: List[int] = field(default_factory=list)


@dataclass
class ReusePlan:
    """Output of collective reuse consumed by Diff-Aware Storage.

    Attributes:
        round_id: The round this plan covers.
        master_req_id: Request chosen as the Master (lowest total deviation).
        group_req_ids: All request IDs in the collective group.
        deviation_scores: Per-request cumulative deviation from the shared
            blocks (lower is better).
        important_positions: Per-request set of token positions that were
            selectively recomputed.
        shared_segment_hashes: Content hashes of segments common to every
            request in the group.
    """

    round_id: str
    master_req_id: str
    group_req_ids: List[str]
    deviation_scores: Dict[str, float]
    important_positions: Dict[str, List[int]]
    shared_segment_hashes: List[int]


# ───────────────────────────────────────────────────────────────────────────
# Compatibility grouping
# ───────────────────────────────────────────────────────────────────────────


def find_compatible_groups(
    requests: Sequence[AgentRequest],
) -> List[List[AgentRequest]]:
    """Partition *requests* into groups eligible for collective reuse.

    Two requests are *compatible* when they belong to the same round,
    have the same active prompt length, the same cached span, and
    non-overlapping slot mappings.

    Args:
        requests: All agent sub-requests currently being scheduled.

    Returns:
        A list of groups, each group a list of compatible
        :class:`AgentRequest` objects.  Singleton groups are included
        (they fall back to the single-request path).
    """
    # Group by (round_id, prompt_len, cached_span)
    buckets: Dict[Tuple[str, int, int], List[AgentRequest]] = {}
    for req in requests:
        key = (req.round_id, req.prompt_len, req.cached_span)
        buckets.setdefault(key, []).append(req)

    groups: List[List[AgentRequest]] = []
    for bucket_key, bucket_reqs in buckets.items():
        # Verify non-overlapping slot mappings within the bucket
        sub_group: List[AgentRequest] = []
        used_slots: Set[int] = set()
        for req in bucket_reqs:
            req_slots = set(req.slot_mapping)
            if req_slots & used_slots:
                # Overlapping — flush current sub-group, start new one
                if sub_group:
                    groups.append(sub_group)
                sub_group = [req]
                used_slots = req_slots
            else:
                sub_group.append(req)
                used_slots |= req_slots
        if sub_group:
            groups.append(sub_group)

    return groups


# ───────────────────────────────────────────────────────────────────────────
# KV Collector
# ───────────────────────────────────────────────────────────────────────────


class KVCollector:
    """Drives collective KV Cache reuse for one group of compatible requests.

    For each compatible group the collector:

    1.  Identifies which token segments are *shared* across all requests
        (using content hashes from the segment index).
    2.  At each transformer layer, concatenates the Q/K tensors from every
        request in the group and applies **one** batched RoPE call.
    3.  On the configured *check layer*, computes key differences between
        rotated cached K and fresh K in **one** batched pass to select
        the *important positions* for the whole group.
    4.  Refreshes only those positions in each request's cached K/V.
    5.  Produces a :class:`ReusePlan` that Diff-Aware Storage consumes.

    The expensive operations — RoPE rotation and key-difference analysis —
    are performed **once** for the group rather than once per request.

    Args:
        num_layers: Number of transformer layers in the model.
        check_layers: Layer indices on which to perform the diff analysis.
        recompute_ratio: Fraction of positions to selectively recompute.
        device: Torch device for tensor operations.
    """

    def __init__(
        self,
        num_layers: int,
        check_layers: Optional[List[int]] = None,
        recompute_ratio: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.check_layers = check_layers or [0]
        self.recompute_ratio = recompute_ratio
        self.device = torch.device(device)

        # Per-group state (set by begin_group, consumed by layer steps)
        self._group: List[AgentRequest] = []
        self._shared_hashes: List[int] = []
        self._imp_positions: Dict[str, List[int]] = {}
        self._deviation_scores: Dict[str, float] = {}

        logger.info(
            "KVCollector: num_layers=%d, check_layers=%s, recompute_ratio=%.2f",
            num_layers,
            self.check_layers,
            recompute_ratio,
        )

    # ── Group lifecycle ─────────────────────────────────────────────────

    def begin_group(self, group: List[AgentRequest]) -> None:
        """Start collective reuse for *group*.

        Must be called before the first :meth:`process_layer` call.

        Args:
            group: The compatible agent requests for this round.
        """
        self._group = group
        self._imp_positions = {}
        self._deviation_scores = {req.req_id: 0.0 for req in group}

        # Identify segments common to *every* request in the group
        if group:
            hash_sets = [set(req.segment_hashes) for req in group]
            common = hash_sets[0]
            for hs in hash_sets[1:]:
                common &= hs
            self._shared_hashes = sorted(common)
        else:
            self._shared_hashes = []

        logger.debug(
            "KVCollector.begin_group: %d requests, %d shared segments",
            len(group),
            len(self._shared_hashes),
        )

    def process_layer(
        self,
        layer_id: int,
        q_tensors: Dict[str, torch.Tensor],
        k_tensors: Dict[str, torch.Tensor],
        v_tensors: Dict[str, torch.Tensor],
        cached_k: Dict[str, torch.Tensor],
        cached_v: Dict[str, torch.Tensor],
        rope_fn: Optional[Any] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Process one transformer layer for the collective group.

        This is the core amortisation step.  On a *check layer*, the method
        performs a single batched difference pass across all requests in the
        group to identify important positions; on all other layers it reuses
        the positions from the check layer.

        Args:
            layer_id: Current layer index.
            q_tensors: ``{req_id: Q}`` freshly computed query tensors.
            k_tensors: ``{req_id: K}`` freshly computed key tensors.
            v_tensors: ``{req_id: V}`` freshly computed value tensors.
            cached_k: ``{req_id: K_cached}`` keys loaded from cache.
            cached_v: ``{req_id: V_cached}`` values loaded from cache.
            rope_fn: Optional callable ``(positions, Q, K) -> (Q', K')``.
            positions: 1-D position tensor for RoPE.

        Returns:
            ``{req_id: (K_updated, V_updated)}`` — the corrected KV pairs
            ready for attention.
        """
        req_ids = [r.req_id for r in self._group]

        # ── 1. Batched RoPE (shared across all requests) ───────────────
        if rope_fn is not None and positions is not None:
            all_q = torch.cat([q_tensors[rid] for rid in req_ids], dim=0)
            all_k = torch.cat([k_tensors[rid] for rid in req_ids], dim=0)
            n_per = q_tensors[req_ids[0]].shape[0]
            batch_pos = positions.repeat(len(req_ids))
            all_q, all_k = rope_fn(batch_pos, all_q, all_k)
            # Scatter back
            for i, rid in enumerate(req_ids):
                q_tensors[rid] = all_q[i * n_per : (i + 1) * n_per]
                k_tensors[rid] = all_k[i * n_per : (i + 1) * n_per]

        # ── 2. Check layer: batched important-position selection ────────
        if layer_id in self.check_layers:
            self._imp_positions = {}
            for rid in req_ids:
                k_new = k_tensors[rid].to(torch.float32)
                k_old = cached_k[rid].to(torch.float32)
                diff = torch.sum((k_new - k_old) ** 2, dim=-1)
                total_len = diff.shape[0]
                topk_num = max(int(total_len * self.recompute_ratio), 1)
                top_idx = torch.topk(diff, k=topk_num).indices
                top_idx_sorted, _ = torch.sort(top_idx)
                self._imp_positions[rid] = top_idx_sorted.tolist()

                # Accumulate deviation scores for master selection
                self._deviation_scores[rid] += diff.sum().item()

        # ── 3. Selective refresh ────────────────────────────────────────
        updated: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for rid in req_ids:
            k_out = cached_k[rid].clone()
            v_out = cached_v[rid].clone()
            if rid in self._imp_positions and self._imp_positions[rid]:
                idx = torch.tensor(
                    self._imp_positions[rid],
                    dtype=torch.long,
                    device=k_out.device,
                )
                k_out[idx] = k_tensors[rid][idx]
                v_out[idx] = v_tensors[rid][idx]
            updated[rid] = (k_out, v_out)

        return updated

    def finish_group(self) -> ReusePlan:
        """Finalize the collective reuse pass and produce a :class:`ReusePlan`.

        The Master is the request with the lowest cumulative deviation from
        the shared blocks.

        Returns:
            A :class:`ReusePlan` consumed by Diff-Aware Storage.
        """
        if not self._group:
            return ReusePlan(
                round_id="",
                master_req_id="",
                group_req_ids=[],
                deviation_scores={},
                important_positions={},
                shared_segment_hashes=[],
            )

        round_id = self._group[0].round_id
        req_ids = [r.req_id for r in self._group]

        # Master = request with lowest deviation
        master_id = min(req_ids, key=lambda rid: self._deviation_scores.get(rid, 0.0))

        plan = ReusePlan(
            round_id=round_id,
            master_req_id=master_id,
            group_req_ids=req_ids,
            deviation_scores=dict(self._deviation_scores),
            important_positions=dict(self._imp_positions),
            shared_segment_hashes=list(self._shared_hashes),
        )

        logger.info(
            "KVCollector.finish_group: round=%s, master=%s, group_size=%d, "
            "shared_segments=%d",
            round_id,
            master_id,
            len(req_ids),
            len(self._shared_hashes),
        )

        # Reset per-group state
        self._group = []
        self._shared_hashes = []
        self._imp_positions = {}
        self._deviation_scores = {}

        return plan
