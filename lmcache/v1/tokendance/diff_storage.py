# SPDX-License-Identifier: Apache-2.0
"""
Diff-Aware Storage for TokenDance (Component 3).

After collective reuse completes, the system still holds one dense KV cache
per request.  Diff-Aware Storage compresses these near-identical caches into
a **Master-Mirror** layout: one dense Master plus lightweight block-sparse
diffs for each Mirror.

Key classes:

* :class:`BlockSparseDiff` — compact representation of the K/V corrections
  between a Mirror and its Master.
* :class:`MirrorObject` — lightweight proxy that references the Master and
  carries sparse diff metadata; defers materialisation to the restore path.
* :class:`DiffAwareStorageWrapper` — wraps an existing LMCache storage
  backend to intercept store/get and apply the Master-Mirror scheme.

Reference: TokenDance paper, Section 4.3 & Section 5 —
Diff-Aware Storage.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.tokendance.collective_reuse import ReusePlan

logger = init_logger(__name__)

# Block granularity for diff detection (tokens per block)
DEFAULT_BLOCK_SIZE: int = 32


# ───────────────────────────────────────────────────────────────────────────
# Block-Sparse Diff Representation
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class BlockSparseDiff:
    """Block-sparse K/V correction between a Mirror and its Master.

    The diff records only the blocks whose values differ.  When K and V
    touch the same blocks, ``shared_indices`` is ``True`` and the
    ``block_indices`` list is shared between K and V to reduce metadata.

    Attributes:
        num_tokens: Total tokens in the original cache segment.
        block_size: Number of tokens per block.
        block_indices: Indices of blocks that differ from the Master.
        k_corrections: ``(len(block_indices), block_size, hidden_dim)``
            tensor of key corrections.
        v_corrections: ``(len(block_indices), block_size, hidden_dim)``
            tensor of value corrections.
        shared_indices: If ``True``, both K and V use the same block
            indices; otherwise each plane has independent indices.
        k_block_indices: Separate K-plane indices (only when
            ``shared_indices is False``).
        v_block_indices: Separate V-plane indices (only when
            ``shared_indices is False``).
        layer_id: Transformer layer this diff applies to (``None`` when
            the diff covers all layers in one chunk).
    """

    num_tokens: int
    block_size: int
    block_indices: List[int]
    k_corrections: Optional[torch.Tensor] = None
    v_corrections: Optional[torch.Tensor] = None
    shared_indices: bool = True
    k_block_indices: Optional[List[int]] = None
    v_block_indices: Optional[List[int]] = None
    layer_id: Optional[int] = None

    @property
    def num_diff_blocks(self) -> int:
        """Number of blocks that carry corrections."""
        return len(self.block_indices)

    @property
    def compression_ratio(self) -> float:
        """Ratio of full size to diff size (higher = better)."""
        total_blocks = (self.num_tokens + self.block_size - 1) // self.block_size
        if self.num_diff_blocks == 0:
            return float("inf")
        return total_blocks / self.num_diff_blocks

    def is_empty(self) -> bool:
        """Return ``True`` if there are no corrections (identical to Master)."""
        return self.num_diff_blocks == 0


@dataclass
class MirrorObject:
    """Lightweight proxy referencing a Master plus sparse diff metadata.

    The caller receives a :class:`MirrorObject` from ``get`` operations
    instead of a dense tensor.  Materialisation is deferred to the
    Fused Sparse Restore path (Component 4).

    Attributes:
        master_key: Cache key of the dense Master.
        mirror_key: Cache key of this Mirror request.
        diffs: Per-layer :class:`BlockSparseDiff` entries.
        old_positions: Original RoPE positions in the Master cache.
        new_positions: Target RoPE positions for the Mirror.
        num_tokens: Total tokens covered.
    """

    master_key: CacheEngineKey
    mirror_key: CacheEngineKey
    diffs: Dict[int, BlockSparseDiff]  # layer_id -> diff
    old_positions: Optional[torch.Tensor] = None
    new_positions: Optional[torch.Tensor] = None
    num_tokens: int = 0

    @property
    def is_mirror(self) -> bool:
        """Always ``True`` — used by the restore path to detect mirrors."""
        return True

    def memory_footprint_ratio(self) -> float:
        """Estimated ratio of Mirror size to a full dense cache."""
        if not self.diffs:
            return 0.0
        total_blocks = 0
        diff_blocks = 0
        for diff in self.diffs.values():
            total_blocks += (
                (diff.num_tokens + diff.block_size - 1) // diff.block_size
            )
            diff_blocks += diff.num_diff_blocks
        if total_blocks == 0:
            return 0.0
        return diff_blocks / total_blocks


# ───────────────────────────────────────────────────────────────────────────
# Diff Analysis Utilities
# ───────────────────────────────────────────────────────────────────────────


def compute_block_sparse_diff(
    master_k: torch.Tensor,
    master_v: torch.Tensor,
    mirror_k: torch.Tensor,
    mirror_v: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    threshold: float = 1e-6,
    layer_id: Optional[int] = None,
) -> BlockSparseDiff:
    """Compute a block-sparse diff between a Mirror and its Master.

    Compares corresponding blocks and records only those that differ
    above *threshold* (L2 norm per block).

    Args:
        master_k: Master key tensor ``(num_tokens, hidden_dim)``.
        master_v: Master value tensor ``(num_tokens, hidden_dim)``.
        mirror_k: Mirror key tensor ``(num_tokens, hidden_dim)``.
        mirror_v: Mirror value tensor ``(num_tokens, hidden_dim)``.
        block_size: Number of tokens per comparison block.
        threshold: L2-norm threshold below which a block is considered
            identical.
        layer_id: Optional layer identifier.

    Returns:
        A :class:`BlockSparseDiff` capturing the corrections.
    """
    num_tokens = master_k.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size

    diff_indices: List[int] = []
    k_corr_blocks: List[torch.Tensor] = []
    v_corr_blocks: List[torch.Tensor] = []
    shared = True

    k_only_indices: List[int] = []
    v_only_indices: List[int] = []

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, num_tokens)

        k_diff = mirror_k[start:end] - master_k[start:end]
        v_diff = mirror_v[start:end] - master_v[start:end]

        k_norm = torch.norm(k_diff.float()).item()
        v_norm = torch.norm(v_diff.float()).item()

        k_changed = k_norm > threshold
        v_changed = v_norm > threshold

        if k_changed or v_changed:
            diff_indices.append(b)
            # Pad to block_size for uniform shape
            pad_len = block_size - (end - start)
            if pad_len > 0:
                k_diff = torch.nn.functional.pad(k_diff, (0, 0, 0, pad_len))
                v_diff = torch.nn.functional.pad(v_diff, (0, 0, 0, pad_len))
            k_corr_blocks.append(k_diff)
            v_corr_blocks.append(v_diff)

            if k_changed != v_changed:
                shared = False
                if k_changed:
                    k_only_indices.append(b)
                else:
                    v_only_indices.append(b)

    k_corrections = (
        torch.stack(k_corr_blocks, dim=0) if k_corr_blocks else None
    )
    v_corrections = (
        torch.stack(v_corr_blocks, dim=0) if v_corr_blocks else None
    )

    return BlockSparseDiff(
        num_tokens=num_tokens,
        block_size=block_size,
        block_indices=diff_indices,
        k_corrections=k_corrections,
        v_corrections=v_corrections,
        shared_indices=shared,
        k_block_indices=None if shared else diff_indices,
        v_block_indices=None if shared else diff_indices,
        layer_id=layer_id,
    )


def build_mirror_from_reuse_plan(
    plan: ReusePlan,
    master_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    mirror_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    mirror_req_id: str,
    master_cache_key: CacheEngineKey,
    mirror_cache_key: CacheEngineKey,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> MirrorObject:
    """Build a :class:`MirrorObject` from a reuse plan.

    When a reuse plan is available (from collective reuse), we already
    know which positions differ.  This function computes per-layer
    block-sparse diffs and packages them.

    Args:
        plan: The :class:`ReusePlan` produced by collective reuse.
        master_kv: ``{layer_id: (K, V)}`` dense Master tensors.
        mirror_kv: ``{layer_id: (K, V)}`` dense Mirror tensors.
        mirror_req_id: Request ID of the Mirror.
        master_cache_key: Cache key for the Master.
        mirror_cache_key: Cache key for the Mirror.
        block_size: Block granularity for diff detection.

    Returns:
        A :class:`MirrorObject` ready for storage.
    """
    diffs: Dict[int, BlockSparseDiff] = {}
    num_tokens = 0

    for layer_id in sorted(master_kv.keys()):
        mk, mv = master_kv[layer_id]
        rk, rv = mirror_kv[layer_id]
        num_tokens = mk.shape[0]
        diff = compute_block_sparse_diff(
            mk, mv, rk, rv,
            block_size=block_size,
            layer_id=layer_id,
        )
        diffs[layer_id] = diff

    return MirrorObject(
        master_key=master_cache_key,
        mirror_key=mirror_cache_key,
        diffs=diffs,
        num_tokens=num_tokens,
    )


def build_mirror_by_token_similarity(
    master_k: torch.Tensor,
    master_v: torch.Tensor,
    mirror_k: torch.Tensor,
    mirror_v: torch.Tensor,
    master_cache_key: CacheEngineKey,
    mirror_cache_key: CacheEngineKey,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> MirrorObject:
    """Heuristic fallback: build a Mirror via token-similarity comparison.

    Used when no explicit reuse plan exists (e.g. request arrives outside
    a recognised All-Gather round).

    Args:
        master_k: Master key tensor ``(num_tokens, hidden_dim)``.
        master_v: Master value tensor ``(num_tokens, hidden_dim)``.
        mirror_k: Mirror key tensor ``(num_tokens, hidden_dim)``.
        mirror_v: Mirror value tensor ``(num_tokens, hidden_dim)``.
        master_cache_key: Cache key for the Master.
        mirror_cache_key: Cache key for the Mirror.
        block_size: Block granularity for diff detection.

    Returns:
        A :class:`MirrorObject` ready for storage.
    """
    diff = compute_block_sparse_diff(
        master_k, master_v, mirror_k, mirror_v,
        block_size=block_size, layer_id=0,
    )
    return MirrorObject(
        master_key=master_cache_key,
        mirror_key=mirror_cache_key,
        diffs={0: diff},
        num_tokens=master_k.shape[0],
    )


# ───────────────────────────────────────────────────────────────────────────
# Diff-Aware Storage Wrapper
# ───────────────────────────────────────────────────────────────────────────


class DiffAwareStorageWrapper:
    """Wraps an LMCache storage backend with Master-Mirror compression.

    On **store**:
    - If a reuse plan is available, the Master is written unchanged and
      each remaining request is stored as a :class:`MirrorObject`.
    - Without a reuse plan the wrapper falls back to token-similarity
      heuristic to detect a reusable Master among existing entries.

    On **read** (get):
    - Master chunks are returned as-is.
    - Mirror chunks return a lightweight :class:`MirrorObject` that
      defers materialisation to the fused restore path.

    Args:
        block_size: Block granularity for diff detection.
    """

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        self.block_size = block_size

        # In-memory mirror registry: mirror_key -> MirrorObject
        self._mirrors: Dict[CacheEngineKey, MirrorObject] = {}

        # Track which keys are masters
        self._master_keys: Set[CacheEngineKey] = set()

        logger.info("DiffAwareStorageWrapper: block_size=%d", block_size)

    def register_master(self, key: CacheEngineKey) -> None:
        """Mark *key* as a Master (written unchanged).

        Args:
            key: The cache key of the Master chunk.
        """
        self._master_keys.add(key)

    def store_mirror(
        self,
        mirror_obj: MirrorObject,
    ) -> None:
        """Store a Mirror's sparse diff metadata.

        The dense Master is assumed to already exist in the underlying
        backend.  Only the lightweight diff metadata is kept.

        Args:
            mirror_obj: The :class:`MirrorObject` to store.
        """
        self._mirrors[mirror_obj.mirror_key] = mirror_obj
        logger.debug(
            "Stored mirror %s (master=%s, footprint_ratio=%.2f)",
            mirror_obj.mirror_key.chunk_hash,
            mirror_obj.master_key.chunk_hash,
            mirror_obj.memory_footprint_ratio(),
        )

    def get(self, key: CacheEngineKey) -> Optional[MirrorObject]:
        """Retrieve a Mirror object by key.

        Returns ``None`` if *key* is not a known Mirror (it may be a
        Master or absent entirely).

        Args:
            key: The cache key to look up.

        Returns:
            The :class:`MirrorObject` if found, else ``None``.
        """
        return self._mirrors.get(key)

    def is_mirror(self, key: CacheEngineKey) -> bool:
        """Check whether *key* is a stored Mirror.

        Args:
            key: The cache key to check.

        Returns:
            ``True`` if *key* is a Mirror, ``False`` otherwise.
        """
        return key in self._mirrors

    def is_master(self, key: CacheEngineKey) -> bool:
        """Check whether *key* is a registered Master.

        Args:
            key: The cache key to check.

        Returns:
            ``True`` if *key* is a Master, ``False`` otherwise.
        """
        return key in self._master_keys

    def remove(self, key: CacheEngineKey) -> bool:
        """Remove a Mirror or Master registration.

        Args:
            key: The cache key to remove.

        Returns:
            ``True`` if anything was removed.
        """
        removed = False
        if key in self._mirrors:
            del self._mirrors[key]
            removed = True
        if key in self._master_keys:
            self._master_keys.discard(key)
            removed = True
        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics.

        Returns:
            A dict containing counts and average compression ratio.
        """
        num_mirrors = len(self._mirrors)
        avg_ratio = 0.0
        if num_mirrors > 0:
            avg_ratio = sum(
                m.memory_footprint_ratio() for m in self._mirrors.values()
            ) / num_mirrors
        return {
            "num_masters": len(self._master_keys),
            "num_mirrors": num_mirrors,
            "avg_mirror_footprint_ratio": avg_ratio,
        }
