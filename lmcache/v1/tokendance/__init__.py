# SPDX-License-Identifier: Apache-2.0
"""
TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing.

This package implements the four TokenDance components on top of LMCache:

1. Round-Aware Segment Indexing  (segment_index, prompt_interface)
2. Collective KV Cache Reuse     (collective_reuse)
3. Diff-Aware Storage            (diff_storage)
4. Fused Sparse Restore          (fused_restore)

Reference: Bian et al., "TokenDance: Scaling Multi-Agent LLM Serving
via Collective KV Cache Sharing", 2026.
"""

from lmcache.v1.tokendance.collective_reuse import (
    AgentRequest,
    KVCollector,
    ReusePlan,
    find_compatible_groups,
)
from lmcache.v1.tokendance.diff_storage import (
    BlockSparseDiff,
    DiffAwareStorageWrapper,
    MirrorObject,
    build_mirror_by_token_similarity,
    build_mirror_from_reuse_plan,
    compute_block_sparse_diff,
)
from lmcache.v1.tokendance.fused_restore import (
    FusedDiffRestorer,
    apply_kv_diff_kernel,
    apply_paired_kv_diff_kernel,
    rope_recover,
)
from lmcache.v1.tokendance.prompt_interface import (
    TTSEP_DEFAULT_STR,
    RoundAwarePromptBuilder,
)
from lmcache.v1.tokendance.segment_index import (
    RoundAwareSegmentDatabase,
    SegmentHashTable,
)

__all__ = [
    # Component 1: Round-Aware Segment Indexing
    "TTSEP_DEFAULT_STR",
    "RoundAwarePromptBuilder",
    "RoundAwareSegmentDatabase",
    "SegmentHashTable",
    # Component 2: Collective KV Cache Reuse
    "AgentRequest",
    "KVCollector",
    "ReusePlan",
    "find_compatible_groups",
    # Component 3: Diff-Aware Storage
    "BlockSparseDiff",
    "DiffAwareStorageWrapper",
    "MirrorObject",
    "build_mirror_by_token_similarity",
    "build_mirror_from_reuse_plan",
    "compute_block_sparse_diff",
    # Component 4: Fused Sparse Restore
    "FusedDiffRestorer",
    "apply_kv_diff_kernel",
    "apply_paired_kv_diff_kernel",
    "rope_recover",
]
