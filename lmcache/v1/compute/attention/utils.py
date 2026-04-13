# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# FlashAttn is always available; FlashInfer sparse may not be.
# Local
from .flash_attn import LMCFlashAttnBackend

try:
    from .flash_infer_sparse import LMCFlashInferSparseBackend

    _FLASHINFER_SPARSE_AVAILABLE = True
except ImportError as _exc:
    _FLASHINFER_SPARSE_AVAILABLE = False
    logger.info(
        "FlashInfer sparse backend not available (%s). "
        "Sparse restore will fall back to FlashAttention.",
        _exc,
    )
    LMCFlashInferSparseBackend = None  # type: ignore[assignment,misc]


def infer_attn_backend_from_vllm(vllm_attn, enable_sparse=False):
    """Select the LMCache attention backend for the given vLLM attention layer.

    On platforms where the FlashInfer block-sparse API is unavailable
    (e.g. GB10/Blackwell with newer FlashInfer builds), the function
    falls back to the standard FlashAttention backend rather than raising.

    Args:
        vllm_attn: A vLLM ``Attention`` layer instance.
        enable_sparse: Whether sparse restore was requested in config.

    Returns:
        An ``AttentionInterface`` implementation.

    Raises:
        ValueError: If no compatible backend can be found.
    """
    attn_name = type(vllm_attn.impl).__name__

    # Sparse path: only when requested AND the FlashInfer sparse API exists
    if attn_name == "FlashInferImpl" and enable_sparse:
        if _FLASHINFER_SPARSE_AVAILABLE and LMCFlashInferSparseBackend is not None:
            return LMCFlashInferSparseBackend(vllm_attn)
        logger.warning(
            "Sparse restore requested but FlashInfer sparse API is "
            "unavailable. Falling back to FlashAttention backend."
        )

    # Standard FlashAttention path — accepts both FlashAttentionImpl and
    # FlashInferImpl (the latter falls back here when sparse is off or
    # unavailable).
    if attn_name in ("FlashAttentionImpl", "FlashInferImpl"):
        return LMCFlashAttnBackend(vllm_attn)

    raise ValueError(
        f"Attention backend '{attn_name}' is not supported in LMCache. "
        f"enable_sparse={enable_sparse}, "
        f"flashinfer_sparse_available={_FLASHINFER_SPARSE_AVAILABLE}"
    )
