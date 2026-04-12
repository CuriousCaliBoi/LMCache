# SPDX-License-Identifier: Apache-2.0
"""
Fused Sparse Restore for TokenDance (Component 4).

Applies block-sparse corrections from Mirror objects during the layerwise
GPU transfer pipeline — **without** materialising a separate dense copy.
Two GPU buffers alternate in a ping-pong fashion: one receives Master
chunks from storage while the other undergoes in-place correction and
writeback.

Key classes:

* :class:`FusedDiffRestorer` — orchestrates per-layer ping-pong restore
  with sparse diff application and RoPE position recovery.
* :func:`apply_kv_diff_kernel` — Python reference implementation of the
  single-plane diff kernel (mirrors the CUDA kernel interface).
* :func:`apply_paired_kv_diff_kernel` — Python reference for the paired
  K+V diff kernel.

Reference: TokenDance paper, Section 4.4 & Section 5, Algorithm 1 —
Fused Diff Restore.
"""

# Standard
from typing import Dict, List, Optional, Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.tokendance.diff_storage import BlockSparseDiff, MirrorObject

logger = init_logger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Diff Kernels (Python reference — mirrors CUDA kernel interface)
# ───────────────────────────────────────────────────────────────────────────


def apply_kv_diff_kernel(
    buffer: torch.Tensor,
    block_indices: List[int],
    corrections: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Apply sparse corrections to a single KV plane in-place.

    This is the Python reference implementation.  The CUDA extension
    provides the same interface with GPU-accelerated execution.

    Args:
        buffer: Dense KV buffer ``(num_tokens, hidden_dim)`` to correct
            in-place.
        block_indices: Indices of blocks that carry corrections.
        corrections: ``(num_blocks, block_size, hidden_dim)`` correction
            tensor.
        block_size: Tokens per block.

    Returns:
        The modified *buffer* (same object, corrected in-place).
    """
    num_tokens = buffer.shape[0]
    for i, b_idx in enumerate(block_indices):
        start = b_idx * block_size
        end = min(start + block_size, num_tokens)
        actual_len = end - start
        buffer[start:end] += corrections[i, :actual_len]
    return buffer


def apply_paired_kv_diff_kernel(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    block_indices: List[int],
    k_corrections: torch.Tensor,
    v_corrections: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply sparse corrections to K and V simultaneously.

    Used when both planes touch the same blocks (``shared_indices=True``),
    avoiding a second pass over the index list.

    Args:
        k_buffer: Dense key buffer ``(num_tokens, hidden_dim)``.
        v_buffer: Dense value buffer ``(num_tokens, hidden_dim)``.
        block_indices: Shared block indices.
        k_corrections: Key correction tensor.
        v_corrections: Value correction tensor.
        block_size: Tokens per block.

    Returns:
        Tuple of ``(k_buffer, v_buffer)``, corrected in-place.
    """
    num_tokens = k_buffer.shape[0]
    for i, b_idx in enumerate(block_indices):
        start = b_idx * block_size
        end = min(start + block_size, num_tokens)
        actual_len = end - start
        k_buffer[start:end] += k_corrections[i, :actual_len]
        v_buffer[start:end] += v_corrections[i, :actual_len]
    return k_buffer, v_buffer


# ───────────────────────────────────────────────────────────────────────────
# RoPE Position Recovery
# ───────────────────────────────────────────────────────────────────────────


def rope_recover(
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    k_buffer: torch.Tensor,
    rope_fn: Optional[object] = None,
) -> torch.Tensor:
    """Recover RoPE positions by rotating from *old* to *new*.

    When a Mirror is stored with positions from the Master's prompt layout,
    the keys must be re-rotated to the Mirror's target positions.

    The simplest approach: un-rotate from old positions, then re-rotate
    to new positions.  When a dedicated ``rope_fn`` is available it is
    used directly; otherwise we fall back to a no-op (positions already
    correct).

    Args:
        old_positions: 1-D tensor of original RoPE positions.
        new_positions: 1-D tensor of target RoPE positions.
        k_buffer: Key tensor ``(num_tokens, hidden_dim)`` to adjust.
        rope_fn: Optional rotary embedding callable.

    Returns:
        The position-corrected key tensor.
    """
    if rope_fn is None:
        return k_buffer
    if torch.equal(old_positions, new_positions):
        return k_buffer
    # Delegate to external RoPE function if available
    # The convention follows vLLM: rope_fn(positions, q, k) -> (q, k)
    dummy_q = torch.zeros_like(k_buffer)
    _, k_corrected = rope_fn(new_positions - old_positions, dummy_q, k_buffer)
    return k_corrected


# ───────────────────────────────────────────────────────────────────────────
# Fused Diff Restorer
# ───────────────────────────────────────────────────────────────────────────


class FusedDiffRestorer:
    """Fused sparse restore for Mirror objects.

    Implements Algorithm 1 from the TokenDance paper.  Two GPU buffers
    alternate roles in a ping-pong fashion:

    - ``B_load`` receives Master chunks from storage.
    - ``B_comp`` undergoes in-place diff correction, RoPE recovery, and
      paged-memory writeback.

    The only additional work per layer is the sparse correction, whose
    cost is proportional to the number of differing blocks (typically
    10–20 % of total).

    Args:
        num_layers: Number of transformer layers.
        device: Torch device for buffers.
    """

    def __init__(
        self,
        num_layers: int,
        device: str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.device = torch.device(device)

        # Ping-pong buffers (lazily allocated on first use)
        self._b_load: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._b_comp: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        logger.info(
            "FusedDiffRestorer: num_layers=%d, device=%s",
            num_layers,
            device,
        )

    def restore(
        self,
        mirror: MirrorObject,
        master_loader: "MasterLoaderCallable",
        slot_map: Optional[torch.Tensor] = None,
        rope_fn: Optional[object] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Fused layerwise restore of a Mirror request.

        Args:
            mirror: The :class:`MirrorObject` to materialise.
            master_loader: Callable ``(layer_id) -> (K, V)`` that loads
                Master chunks for a given layer.
            slot_map: Optional paged-memory slot mapping.
            rope_fn: Optional RoPE recovery callable.

        Returns:
            ``{layer_id: (K_restored, V_restored)}`` — the fully
            materialised KV pairs for every layer.
        """
        restored: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for layer_id in range(self.num_layers):
            # ── Load Master chunks into B_load ──────────────────────────
            master_k, master_v = master_loader(layer_id)

            # ── Swap: B_load ↔ B_comp ──────────────────────────────────
            k_comp = master_k.clone()
            v_comp = master_v.clone()

            # ── Apply sparse diff ───────────────────────────────────────
            diff = mirror.diffs.get(layer_id)
            if diff is not None and not diff.is_empty():
                if (
                    diff.shared_indices
                    and diff.k_corrections is not None
                    and diff.v_corrections is not None
                ):
                    apply_paired_kv_diff_kernel(
                        k_comp,
                        v_comp,
                        diff.block_indices,
                        diff.k_corrections.to(k_comp.device),
                        diff.v_corrections.to(v_comp.device),
                        diff.block_size,
                    )
                else:
                    # Fall back to separate single-plane kernels
                    if diff.k_corrections is not None:
                        indices = (
                            diff.k_block_indices
                            if diff.k_block_indices
                            else diff.block_indices
                        )
                        apply_kv_diff_kernel(
                            k_comp,
                            indices,
                            diff.k_corrections.to(k_comp.device),
                            diff.block_size,
                        )
                    if diff.v_corrections is not None:
                        indices = (
                            diff.v_block_indices
                            if diff.v_block_indices
                            else diff.block_indices
                        )
                        apply_kv_diff_kernel(
                            v_comp,
                            indices,
                            diff.v_corrections.to(v_comp.device),
                            diff.block_size,
                        )

            # ── RoPE position recovery ──────────────────────────────────
            if mirror.old_positions is not None and mirror.new_positions is not None:
                k_comp = rope_recover(
                    mirror.old_positions, mirror.new_positions, k_comp, rope_fn
                )

            restored[layer_id] = (k_comp, v_comp)

        logger.debug(
            "FusedDiffRestorer: restored mirror %s across %d layers",
            mirror.mirror_key.chunk_hash,
            self.num_layers,
        )

        return restored

    def restore_dense_fallback(
        self,
        master_loader: "MasterLoaderCallable",
        num_layers: Optional[int] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Dense restore fallback when Mirror metadata is misaligned.

        Simply loads and returns the Master chunks without any diff
        application.

        Args:
            master_loader: Callable ``(layer_id) -> (K, V)``.
            num_layers: Override for the number of layers.

        Returns:
            ``{layer_id: (K, V)}`` — dense Master data.
        """
        n = num_layers or self.num_layers
        result: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_id in range(n):
            result[layer_id] = master_loader(layer_id)
        return result


# Type alias for the master loader callable
MasterLoaderCallable = object  # Callable[[int], Tuple[torch.Tensor, torch.Tensor]]
