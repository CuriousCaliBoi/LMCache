# SPDX-License-Identifier: Apache-2.0
"""
TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing.

This package implements the four TokenDance components on top of LMCache:

1. Round-Aware Segment Indexing  (segment_index)
2. Collective KV Cache Reuse     (collective_reuse)
3. Diff-Aware Storage            (diff_storage)
4. Fused Sparse Restore          (fused_restore)

Reference: Bian et al., "TokenDance: Scaling Multi-Agent LLM Serving
via Collective KV Cache Sharing", 2026.
"""

from lmcache.v1.tokendance.prompt_interface import (
    TTSEP_DEFAULT_STR,
    RoundAwarePromptBuilder,
)
from lmcache.v1.tokendance.segment_index import (
    RoundAwareSegmentDatabase,
    SegmentHashTable,
)

__all__ = [
    "TTSEP_DEFAULT_STR",
    "RoundAwarePromptBuilder",
    "RoundAwareSegmentDatabase",
    "SegmentHashTable",
]
