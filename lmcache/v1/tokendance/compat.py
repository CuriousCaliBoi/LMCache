# SPDX-License-Identifier: Apache-2.0
"""
TokenDance / LMCache compatibility patch for vLLM ≥ 0.19 on DGX Spark (GB10).

This module addresses three integration issues between the tokendance-impl
branch and vllm 0.19.1rc1.dev228+g4beeb0689.cu130:

1. VLLMModelTracker.register_model() — proper hook point so sitecustomize.py
   can be removed.
2. FlashInfer sparse backend gating — prevents import failures when
   FlashInfer's block-sparse API is unavailable or incompatible.
3. vLLM import-path resilience — try/except wrappers for paths that
   moved across vLLM versions.
"""

# Standard
from typing import Optional

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def safe_register_vllm_model(instance_id: str, vllm_model: nn.Module) -> bool:
    """Register a vLLM model with VLLMModelTracker, handling import failures.

    This is the proper call site replacement for any ``sitecustomize.py``
    hook.  It should be called from the worker-side connector initialization,
    after the model is loaded but **before** ``LMCBlenderBuilder.get_or_create``.

    Args:
        instance_id: The LMCache engine instance ID (typically ``ENGINE_NAME``).
        vllm_model: The loaded vLLM model (e.g. ``LlamaForCausalLM``).

    Returns:
        ``True`` if registration succeeded, ``False`` on failure.
    """
    try:
        # First Party
        from lmcache.v1.compute.models.utils import VLLMModelTracker

        VLLMModelTracker.register_model(instance_id, vllm_model)
        logger.info(
            "Registered vLLM model '%s' for instance '%s'",
            type(vllm_model).__name__,
            instance_id,
        )
        return True
    except Exception as exc:
        logger.warning(
            "Failed to register vLLM model for blending: %s. "
            "Blending will be unavailable.",
            exc,
        )
        return False


def is_flashinfer_sparse_available() -> bool:
    """Check whether the FlashInfer block-sparse attention API is usable.

    On GB10/Blackwell with newer FlashInfer builds, the
    ``VariableBlockSparseAttentionWrapper`` and related page-level APIs
    may be absent or incompatible.  This function probes availability
    so callers can gate the sparse backend cleanly.

    Returns:
        ``True`` if all required FlashInfer sparse APIs are importable.
    """
    try:
        from flashinfer import VariableBlockSparseAttentionWrapper  # noqa: F401
        from flashinfer.page import (  # noqa: F401
            block_sparse_indices_to_vector_sparse_offsets,
        )
        from flashinfer.utils import (  # noqa: F401
            TensorLayout,
            _check_pos_encoding_mode,
            check_shape_dtype_device,
            device_support_pdl,
        )

        return True
    except (ImportError, AttributeError) as exc:
        logger.info(
            "FlashInfer sparse attention API not available: %s. "
            "Sparse restore backend will be disabled.",
            exc,
        )
        return False


def safe_infer_attn_backend(vllm_attn: object, enable_sparse: bool = False):
    """Infer the LMCache attention backend with FlashInfer-sparse gating.

    Drop-in replacement for
    ``lmcache.v1.compute.attention.utils.infer_attn_backend_from_vllm``
    that gates the FlashInfer sparse backend behind an availability check.

    Args:
        vllm_attn: The vLLM ``Attention`` layer.
        enable_sparse: Whether sparse mode was requested in config.

    Returns:
        An ``AttentionInterface`` instance.

    Raises:
        ValueError: If no compatible backend is found.
    """
    attn_name = type(vllm_attn.impl).__name__  # type: ignore[union-attr]

    if attn_name == "FlashInferImpl" and enable_sparse:
        if is_flashinfer_sparse_available():
            from lmcache.v1.compute.attention.flash_infer_sparse import (
                LMCFlashInferSparseBackend,
            )

            return LMCFlashInferSparseBackend(vllm_attn)
        else:
            logger.warning(
                "Sparse restore requested but FlashInfer sparse API "
                "is unavailable. Falling back to FlashAttention backend."
            )
            # Fall through to FlashAttn path

    if attn_name in ("FlashAttentionImpl", "FlashInferImpl"):
        try:
            from lmcache.v1.compute.attention.flash_attn import (
                LMCFlashAttnBackend,
            )

            return LMCFlashAttnBackend(vllm_attn)
        except ImportError as exc:
            logger.warning("FlashAttention backend unavailable: %s", exc)

    raise ValueError(
        f"No compatible LMCache attention backend for {attn_name}. "
        f"enable_sparse={enable_sparse}, "
        f"flashinfer_sparse_available={is_flashinfer_sparse_available()}"
    )
