# SPDX-License-Identifier: Apache-2.0
"""
Round-Aware Segment Indexing for TokenDance.

This module replaces the fixed-size chunk hash table used by
:class:`ChunkedTokenDatabase` with a **segment-based hash table** that
splits prompts at ``<TTSEP>`` separator boundaries and indexes each
segment independently by its *content hash* (position-independent).

Two requests that contain the same shared update therefore map that update
to the same cache object, even when their private histories differ in
length.

Key classes:

* :class:`SegmentHashTable` — an in-memory index mapping
  ``segment_content_hash  →  segment_id``.
* :class:`RoundAwareSegmentDatabase` — a :class:`TokenDatabase` subclass
  that splits prompts at ``<TTSEP>``, hashes each segment by content,
  and emits one ``CacheEngineKey`` per segment.

Reference: TokenDance paper, Section 4.1 & Section 5 —
Round-Aware Segment Indexing.
"""

# Standard
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import (
    NONE_HASH,
    ProcessTokensResult,
    TokenDatabase,
)
from lmcache.v1.tokendance.prompt_interface import TTSEP_DEFAULT_STR

logger = init_logger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Segment Hash Table
# ───────────────────────────────────────────────────────────────────────────


class SegmentHashTable:
    """Content-addressed index for variable-length token segments.

    Unlike the fixed-size rolling-hash table in
    :class:`ChunkedTokenDatabase`, this table indexes segments of arbitrary
    length identified by their **content hash**.  The hash depends only on
    the tokens inside the segment — *not* on the absolute position of the
    segment in the prompt — so two requests that share a block will
    produce the same hash and hit the same cache entry.

    The table is thread-safe; all mutations are guarded by a lock.

    Args:
        initial_capacity: Expected number of unique segments (used only
            for internal sizing hints — the table grows dynamically).
    """

    def __init__(self, initial_capacity: int = 4096) -> None:
        self._lock = threading.Lock()

        # segment_content_hash  →  segment_id (monotonic counter)
        self._hash_to_id: Dict[int, int] = {}

        # segment_id  →  segment_content_hash  (reverse map for eviction)
        self._id_to_hash: Dict[int, int] = {}

        # Monotonic counter for segment IDs
        self._next_id: int = 0

        # Track access timestamps for statistics / eviction hints
        self._access_ts: Dict[int, float] = {}

        logger.info(
            "SegmentHashTable created (initial_capacity=%d)", initial_capacity
        )

    # ── Queries ──────────────────────────────────────────────────────────

    def lookup(self, content_hash: int) -> Optional[int]:
        """Return the segment ID for *content_hash*, or ``None`` on miss.

        Args:
            content_hash: The content-based hash of a token segment.

        Returns:
            The integer segment ID if found, else ``None``.
        """
        with self._lock:
            sid = self._hash_to_id.get(content_hash)
            if sid is not None:
                self._access_ts[sid] = time.monotonic()
            return sid

    def batch_lookup(self, content_hashes: List[int]) -> List[Optional[int]]:
        """Vectorised lookup for multiple hashes.

        Args:
            content_hashes: List of content hashes to look up.

        Returns:
            List of segment IDs (``None`` where the hash is absent).
        """
        now = time.monotonic()
        with self._lock:
            results: List[Optional[int]] = []
            for h in content_hashes:
                sid = self._hash_to_id.get(h)
                if sid is not None:
                    self._access_ts[sid] = now
                results.append(sid)
            return results

    def contains(self, content_hash: int) -> bool:
        """Check whether *content_hash* is in the table.

        Args:
            content_hash: The content-based hash of a token segment.

        Returns:
            ``True`` if the hash is indexed, ``False`` otherwise.
        """
        with self._lock:
            return content_hash in self._hash_to_id

    # ── Mutations ────────────────────────────────────────────────────────

    def insert(self, content_hash: int) -> int:
        """Insert a new segment or return the existing ID.

        If *content_hash* is already present the existing segment ID is
        returned without mutation.

        Args:
            content_hash: The content-based hash of the segment.

        Returns:
            The (possibly new) integer segment ID.
        """
        now = time.monotonic()
        with self._lock:
            existing = self._hash_to_id.get(content_hash)
            if existing is not None:
                self._access_ts[existing] = now
                return existing
            sid = self._next_id
            self._next_id += 1
            self._hash_to_id[content_hash] = sid
            self._id_to_hash[sid] = content_hash
            self._access_ts[sid] = now
            return sid

    def batch_insert(self, content_hashes: List[int]) -> List[int]:
        """Insert multiple hashes in one locked critical section.

        Args:
            content_hashes: List of content hashes.

        Returns:
            List of segment IDs (one per input hash).
        """
        now = time.monotonic()
        ids: List[int] = []
        with self._lock:
            for h in content_hashes:
                existing = self._hash_to_id.get(h)
                if existing is not None:
                    self._access_ts[existing] = now
                    ids.append(existing)
                else:
                    sid = self._next_id
                    self._next_id += 1
                    self._hash_to_id[h] = sid
                    self._id_to_hash[sid] = h
                    self._access_ts[sid] = now
                    ids.append(sid)
        return ids

    def evict(self, content_hash: int) -> bool:
        """Remove a segment from the table.

        Args:
            content_hash: Hash of the segment to remove.

        Returns:
            ``True`` if the segment was present and removed, ``False``
            otherwise.
        """
        with self._lock:
            sid = self._hash_to_id.pop(content_hash, None)
            if sid is not None:
                self._id_to_hash.pop(sid, None)
                self._access_ts.pop(sid, None)
                return True
            return False

    # ── Statistics ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of indexed segments."""
        with self._lock:
            return len(self._hash_to_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics.

        Returns:
            A dict containing ``num_segments`` and ``next_id``.
        """
        with self._lock:
            return {
                "num_segments": len(self._hash_to_id),
                "next_id": self._next_id,
            }

    def clear(self) -> None:
        """Remove all entries from the table."""
        with self._lock:
            self._hash_to_id.clear()
            self._id_to_hash.clear()
            self._access_ts.clear()
            self._next_id = 0


# ───────────────────────────────────────────────────────────────────────────
# Round-Aware Segment Token Database
# ───────────────────────────────────────────────────────────────────────────


class RoundAwareSegmentDatabase(TokenDatabase):
    """Token database that splits prompts at ``<TTSEP>`` and indexes
    segments by content hash.

    This is the runtime counterpart of
    :class:`RoundAwarePromptBuilder`.  Where the builder *inserts*
    separators, this database *splits* on them and produces one
    ``CacheEngineKey`` per segment whose ``chunk_hash`` is derived solely
    from the segment's token content — making it position-independent.

    Two agents whose prompts share the same output block will produce
    the same content hash for that block, regardless of where it appears
    in each prompt.  The :class:`SegmentHashTable` ensures that the cache
    engine stores the block exactly once.

    Args:
        config: The LMCache engine configuration.
        metadata: Runtime metadata (model name, world size, etc.).
        sep_str: Separator string.  Defaults to ``"<TTSEP>"``.
        segment_hash_table: An optional pre-existing table; one is created
            automatically if omitted.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        sep_str: Optional[str] = None,
        segment_hash_table: Optional[SegmentHashTable] = None,
    ) -> None:
        super().__init__(config, metadata)

        # Determine separator string — respect config if available,
        # otherwise fall back to the explicit argument or default.
        if sep_str is not None:
            self._sep_str = sep_str
        elif hasattr(config, "tokendance_sep_str") and config.tokendance_sep_str:
            self._sep_str = config.tokendance_sep_str
        elif config.blend_special_str:
            self._sep_str = config.blend_special_str
        else:
            self._sep_str = TTSEP_DEFAULT_STR

        # Tokenize the separator using the model's tokenizer.
        # Third Party
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
        raw_ids: list[int] = self._tokenizer.encode(self._sep_str)
        # Strip leading BOS if the tokenizer prepends one
        if (
            hasattr(self._tokenizer, "bos_token_id")
            and len(raw_ids) > 1
            and raw_ids[0] == self._tokenizer.bos_token_id
        ):
            raw_ids = raw_ids[1:]
        self._sep_ids: torch.Tensor = torch.tensor(
            raw_ids, dtype=torch.long, device="cpu"
        )
        self._sep_len: int = len(raw_ids)

        # Segment hash table
        self._segment_table: SegmentHashTable = (
            segment_hash_table
            if segment_hash_table is not None
            else SegmentHashTable()
        )

        logger.info(
            "RoundAwareSegmentDatabase: sep_str=%r  sep_ids=%s  sep_len=%d",
            self._sep_str,
            raw_ids,
            self._sep_len,
        )

    # ── TokenDatabase interface ─────────────────────────────────────────

    @_lmcache_nvtx_annotate
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[ProcessTokensResult]:
        """Split the token stream at ``<TTSEP>`` boundaries and yield one
        ``(start, end, key_or_hash)`` triple per segment.

        When *tokens* are provided, the method:

        1. Finds all occurrences of the separator token sequence.
        2. Splits the prompt into segments at those boundaries.
        3. Computes a **content-only** hash for each segment (no prefix
           chaining) and inserts it into the segment hash table.
        4. Yields ``(start_idx, end_idx, CacheEngineKey)`` for every
           segment that falls outside the ``mask`` prefix.

        When *hashes* and *offsets* are provided (pre-computed path from
        the multiprocess server), the method directly wraps each hash
        into a ``CacheEngineKey``.

        Args:
            tokens: Full token sequence (1-D tensor or list).
            hashes: Pre-computed content hashes (one per segment).
            offsets: Number of tokens in each segment.
            mask: Boolean mask of length ``len(tokens)``.  Leading
                ``False`` entries mark the prefix that should be skipped.
            make_key: If ``True`` return ``CacheEngineKey``; otherwise
                return the raw hash value.
            request_configs: Per-request metadata forwarded to the key.

        Yields:
            ``(start_index, end_index, key_or_hash)`` tuples.
        """
        if tokens is not None:
            yield from self._process_from_tokens(
                tokens, mask, make_key, request_configs
            )
        elif hashes is not None:
            yield from self._process_from_hashes(
                hashes, offsets, make_key, request_configs
            )
        else:
            raise ValueError("Either tokens or hashes must be provided.")

    # ── Public accessors ────────────────────────────────────────────────

    def get_segment_table(self) -> SegmentHashTable:
        """Return the underlying :class:`SegmentHashTable`.

        Returns:
            The segment hash table instance.
        """
        return self._segment_table

    def get_separator_ids(self) -> torch.Tensor:
        """Return the separator token IDs as a 1-D tensor.

        Returns:
            A ``torch.LongTensor`` with the separator token IDs.
        """
        return self._sep_ids.clone()

    # ── Private: process from raw tokens ────────────────────────────────

    def _process_from_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor],
        make_key: bool,
        request_configs: Optional[dict],
    ) -> Iterable[ProcessTokensResult]:
        """Core splitting / hashing logic for token-based input."""
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long, device="cpu")
        else:
            tokens = tokens.to(device="cpu", dtype=torch.long)

        num_falses = 0
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()

        # Split at separator boundaries
        segments = self._split_at_separators(tokens)

        # Walk through segments, track absolute offsets, emit results
        abs_offset = 0
        for seg_idx, segment_tokens in enumerate(segments):
            seg_len = len(segment_tokens)

            # Compute the absolute start/end of the segment *content*
            # (i.e. excluding the separator that preceded it).
            if seg_idx == 0:
                # First segment — no preceding separator
                start_idx = abs_offset
                end_idx = abs_offset + seg_len
                abs_offset = end_idx
            else:
                # Account for the separator tokens that were consumed
                abs_offset += self._sep_len
                start_idx = abs_offset
                end_idx = abs_offset + seg_len
                abs_offset = end_idx

            # Skip segments entirely within the masked prefix
            if end_idx <= num_falses:
                continue

            # Skip zero-length segments (adjacent separators)
            if seg_len == 0:
                continue

            # Content-based hash — independent of absolute position
            content_hash = self._hash_tokens(segment_tokens)

            # Register in the segment hash table
            self._segment_table.insert(content_hash)

            if make_key:
                yield (
                    start_idx,
                    end_idx,
                    self._make_key_by_hash(content_hash, request_configs),
                )
            else:
                yield start_idx, end_idx, content_hash

    # ── Private: process from pre-computed hashes ───────────────────────

    def _process_from_hashes(
        self,
        hashes: List[int],
        offsets: Optional[List[int]],
        make_key: bool,
        request_configs: Optional[dict],
    ) -> Iterable[ProcessTokensResult]:
        """Wrap pre-computed hashes into ``CacheEngineKey`` objects."""
        assert offsets is not None, (
            "If hashes are provided, offsets must also be provided."
        )
        start_idx = 0
        for hash_val, offset in zip(hashes, offsets, strict=False):
            end_idx = start_idx + offset
            self._segment_table.insert(hash_val)
            if make_key:
                yield (
                    start_idx,
                    end_idx,
                    self._make_key_by_hash(hash_val, request_configs),
                )
            else:
                yield start_idx, end_idx, hash_val
            start_idx = end_idx

    # ── Private: separator-based splitting ──────────────────────────────

    def _split_at_separators(
        self, tokens: torch.Tensor
    ) -> List[torch.Tensor]:
        """Split *tokens* at every occurrence of the separator sequence.

        Returns a list of token tensors — one per segment.  The separator
        tokens themselves are **not** included in any segment.

        Args:
            tokens: 1-D LongTensor to split.

        Returns:
            List of 1-D LongTensors (segments).
        """
        if self._sep_len == 0 or len(tokens) < self._sep_len:
            return [tokens]

        # Sliding-window match — same technique as SegmentTokenDatabase
        windows = tokens.unfold(0, self._sep_len, 1)
        match_mask = (windows == self._sep_ids).all(dim=1)
        match_positions: List[int] = match_mask.nonzero(as_tuple=True)[0].tolist()

        if not match_positions:
            return [tokens]

        segments: List[torch.Tensor] = []
        start = 0
        for pos in match_positions:
            segments.append(tokens[start:pos])
            start = pos + self._sep_len
        # Trailing segment after last separator
        segments.append(tokens[start:])
        return segments
