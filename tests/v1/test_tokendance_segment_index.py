# SPDX-License-Identifier: Apache-2.0
"""
Tests for TokenDance Round-Aware Segment Indexing.

Covers:
- SegmentHashTable: insert, lookup, batch ops, eviction, stats, thread safety
- RoundAwareSegmentDatabase: separator splitting, content hashing,
  position-independence, mask handling, hash/offset path
- RoundAwarePromptBuilder: prompt construction, separator injection
- Integration: two agents sharing blocks produce identical segment hashes
"""

# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.tokendance.prompt_interface import (
    TTSEP_DEFAULT_STR,
    RoundAwarePromptBuilder,
)
from lmcache.v1.tokendance.segment_index import (
    RoundAwareSegmentDatabase,
    SegmentHashTable,
)

# Local
from .utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens


# ───────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def segment_table():
    """A fresh SegmentHashTable for each test."""
    return SegmentHashTable(initial_capacity=128)


# ───────────────────────────────────────────────────────────────────────────
# SegmentHashTable unit tests
# ───────────────────────────────────────────────────────────────────────────


class TestSegmentHashTable:
    """Tests for the content-addressed segment hash table."""

    def test_insert_and_lookup(self, segment_table):
        """Inserting a hash should return a stable segment ID."""
        sid = segment_table.insert(42)
        assert sid == 0  # first insert gets ID 0
        assert segment_table.lookup(42) == 0

    def test_duplicate_insert_returns_same_id(self, segment_table):
        """Re-inserting the same hash must return the original ID."""
        sid1 = segment_table.insert(100)
        sid2 = segment_table.insert(100)
        assert sid1 == sid2

    def test_different_hashes_get_different_ids(self, segment_table):
        """Distinct hashes must receive distinct IDs."""
        ids = [segment_table.insert(h) for h in range(10)]
        assert len(set(ids)) == 10

    def test_lookup_miss(self, segment_table):
        """Looking up an absent hash returns None."""
        assert segment_table.lookup(9999) is None

    def test_contains(self, segment_table):
        """contains() should reflect insert state."""
        assert not segment_table.contains(7)
        segment_table.insert(7)
        assert segment_table.contains(7)

    def test_batch_insert_and_lookup(self, segment_table):
        """batch_insert and batch_lookup must agree."""
        hashes = [10, 20, 30, 20, 10]
        ids = segment_table.batch_insert(hashes)
        # Duplicates should map to the same IDs
        assert ids[0] == ids[4]  # hash 10
        assert ids[1] == ids[3]  # hash 20
        assert len(set(ids)) == 3  # three unique hashes

        looked_up = segment_table.batch_lookup([30, 10, 999])
        assert looked_up[0] == ids[2]
        assert looked_up[1] == ids[0]
        assert looked_up[2] is None

    def test_evict(self, segment_table):
        """Evicting a hash should remove it from the table."""
        segment_table.insert(55)
        assert segment_table.contains(55)
        removed = segment_table.evict(55)
        assert removed is True
        assert not segment_table.contains(55)
        # Second evict returns False
        assert segment_table.evict(55) is False

    def test_len_and_stats(self, segment_table):
        """__len__ and get_stats should be consistent."""
        segment_table.batch_insert([1, 2, 3])
        assert len(segment_table) == 3
        stats = segment_table.get_stats()
        assert stats["num_segments"] == 3
        assert stats["next_id"] == 3

    def test_clear(self, segment_table):
        """clear() should empty the table."""
        segment_table.batch_insert(list(range(50)))
        assert len(segment_table) == 50
        segment_table.clear()
        assert len(segment_table) == 0
        assert segment_table.lookup(0) is None

    def test_thread_safety(self, segment_table):
        """Concurrent inserts should not corrupt the table."""
        errors: list = []

        def inserter(start: int, count: int):
            try:
                for i in range(start, start + count):
                    segment_table.insert(i)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=inserter, args=(i * 1000, 1000))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(segment_table) == 4000


# ───────────────────────────────────────────────────────────────────────────
# RoundAwareSegmentDatabase unit tests
# ───────────────────────────────────────────────────────────────────────────


def _make_segment_db(
    sep_str: str = " # # ",
    model_name: str = "facebook/opt-125m",
) -> RoundAwareSegmentDatabase:
    """Helper: create a RoundAwareSegmentDatabase with a small model."""
    cfg = LMCacheEngineConfig.from_legacy(
        blend_special_str=sep_str,
        save_unfull_chunk=True,
    )
    meta = dumb_metadata_with_model_name(model_name)
    return RoundAwareSegmentDatabase(cfg, meta, sep_str=sep_str)


def _requires_opt_tokenizer():
    """Skip if facebook/opt-125m cannot be loaded (no HF credentials)."""
    try:
        # Third Party
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("facebook/opt-125m")
        return False
    except Exception:
        return True


_skip_no_tokenizer = pytest.mark.skipif(
    _requires_opt_tokenizer(),
    reason="facebook/opt-125m tokenizer not available",
)


@_skip_no_tokenizer
class TestRoundAwareSegmentDatabase:
    """Tests for the segment-based TokenDatabase."""

    def test_basic_split(self):
        """Segments separated by the separator token produce distinct keys."""
        db = _make_segment_db()
        sep = db.get_separator_ids()

        # Build: [seg_A] <SEP> [seg_B] <SEP> [seg_C]
        seg_a = torch.tensor([10, 20, 30], dtype=torch.long)
        seg_b = torch.tensor([40, 50], dtype=torch.long)
        seg_c = torch.tensor([60, 70, 80, 90], dtype=torch.long)
        tokens = torch.cat([seg_a, sep, seg_b, sep, seg_c])

        results = list(db.process_tokens(tokens=tokens))
        assert len(results) == 3

        # Check spans
        assert results[0][0] == 0
        assert results[0][1] == 3  # seg_a length
        assert results[1][1] - results[1][0] == 2  # seg_b length
        assert results[2][1] - results[2][0] == 4  # seg_c length

    def test_no_separator(self):
        """A prompt with no separator produces one segment."""
        db = _make_segment_db()
        tokens = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        results = list(db.process_tokens(tokens=tokens))
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] == 5

    def test_content_hash_is_position_independent(self):
        """The same token block at different offsets must produce the same
        content hash."""
        db = _make_segment_db()
        sep = db.get_separator_ids()
        sep_len = len(sep)

        shared_block = torch.tensor([100, 200, 300], dtype=torch.long)

        # Agent A: short history + shared block
        history_a = torch.tensor([1, 2], dtype=torch.long)
        prompt_a = torch.cat([history_a, sep, shared_block])

        # Agent B: longer history + same shared block
        history_b = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
        prompt_b = torch.cat([history_b, sep, shared_block])

        results_a = list(db.process_tokens(tokens=prompt_a, make_key=False))
        results_b = list(db.process_tokens(tokens=prompt_b, make_key=False))

        # Both should have 2 segments
        assert len(results_a) == 2
        assert len(results_b) == 2

        # The shared block hash must match despite different absolute offsets
        hash_a_shared = results_a[1][2]
        hash_b_shared = results_b[1][2]
        assert hash_a_shared == hash_b_shared

        # The private history hashes must differ
        assert results_a[0][2] != results_b[0][2]

    def test_mask_skips_prefix(self):
        """Segments within the masked prefix should be omitted."""
        db = _make_segment_db()
        sep = db.get_separator_ids()
        sep_len = len(sep)

        seg_a = torch.tensor([10, 20], dtype=torch.long)
        seg_b = torch.tensor([30, 40], dtype=torch.long)
        seg_c = torch.tensor([50, 60], dtype=torch.long)
        tokens = torch.cat([seg_a, sep, seg_b, sep, seg_c])

        total_len = len(tokens)
        mask = torch.ones(total_len, dtype=torch.bool)
        # Mask out the first segment + separator
        prefix_len = len(seg_a) + sep_len
        mask[:prefix_len] = False

        results = list(db.process_tokens(tokens=tokens, mask=mask))
        # seg_a is masked out → only seg_b and seg_c returned
        assert len(results) == 2
        # First returned segment should be seg_b
        assert results[0][1] - results[0][0] == 2

    def test_hash_offset_path(self):
        """Pre-computed hashes + offsets path must yield correct keys."""
        db = _make_segment_db()
        hashes_in = [111, 222, 333]
        offsets_in = [10, 20, 30]

        results = list(
            db.process_tokens(hashes=hashes_in, offsets=offsets_in)
        )
        assert len(results) == 3
        assert results[0] == (0, 10, results[0][2])
        assert results[1] == (10, 30, results[1][2])
        assert results[2] == (30, 60, results[2][2])

        # All hashes should be in the segment table
        table = db.get_segment_table()
        assert len(table) == 3
        for h in hashes_in:
            assert table.contains(h)

    def test_segment_table_is_populated(self):
        """process_tokens must register every segment in the table."""
        db = _make_segment_db()
        sep = db.get_separator_ids()

        tokens = torch.cat([
            torch.tensor([1, 2], dtype=torch.long),
            sep,
            torch.tensor([3, 4], dtype=torch.long),
            sep,
            torch.tensor([5, 6], dtype=torch.long),
        ])
        list(db.process_tokens(tokens=tokens))

        table = db.get_segment_table()
        assert len(table) == 3

    def test_multi_agent_shared_segment_dedup(self):
        """Two agents sharing a block should produce the same segment ID
        in the hash table (proving deduplication)."""
        table = SegmentHashTable()
        db = _make_segment_db()
        # Override the shared table
        db._segment_table = table  # noqa: SLF001 — test-only

        sep = db.get_separator_ids()
        shared = torch.tensor([100, 200, 300], dtype=torch.long)

        # Agent 1
        prompt1 = torch.cat([
            torch.tensor([1], dtype=torch.long), sep, shared
        ])
        r1 = list(db.process_tokens(tokens=prompt1, make_key=False))

        # Agent 2
        prompt2 = torch.cat([
            torch.tensor([2, 3], dtype=torch.long), sep, shared
        ])
        r2 = list(db.process_tokens(tokens=prompt2, make_key=False))

        # Shared block hashes match
        assert r1[1][2] == r2[1][2]

        # Table should have 3 entries: agent1_history, agent2_history, shared
        assert len(table) == 3

    def test_empty_tokens(self):
        """Empty input should yield nothing."""
        db = _make_segment_db()
        results = list(db.process_tokens(tokens=torch.tensor([], dtype=torch.long)))
        assert len(results) == 0


# ───────────────────────────────────────────────────────────────────────────
# RoundAwarePromptBuilder tests
# ───────────────────────────────────────────────────────────────────────────


@_skip_no_tokenizer
class TestRoundAwarePromptBuilder:
    """Tests for the application-facing prompt builder."""

    def _get_tokenizer(self):
        # Third Party
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("facebook/opt-125m")

    def test_separator_injected(self):
        """The output tensor must contain separator tokens between blocks."""
        tokenizer = self._get_tokenizer()
        builder = RoundAwarePromptBuilder(tokenizer, sep_str=" # # ")
        sep_ids = builder.get_separator_tokens()

        prompt = builder.build_round_prompt(
            private_history_text="Hello",
            shared_blocks=["Block A", "Block B"],
        )
        prompt_list = prompt.tolist()

        # Count separator occurrences
        sep_count = 0
        for i in range(len(prompt_list) - len(sep_ids) + 1):
            if prompt_list[i : i + len(sep_ids)] == sep_ids:
                sep_count += 1
        assert sep_count == 2  # one before each shared block

    def test_round_trip_with_segment_db(self):
        """Prompt built by the builder should be correctly split by the
        database."""
        tokenizer = self._get_tokenizer()
        builder = RoundAwarePromptBuilder(tokenizer, sep_str=" # # ")
        db = _make_segment_db()

        prompt = builder.build_round_prompt(
            private_history_text="System: You are agent X.",
            shared_blocks=["Agent A said hello.", "Agent B said goodbye."],
            round_task_text="What do you reply?",
        )
        results = list(db.process_tokens(tokens=prompt))
        # 1 private + 2 shared + 1 task = 4 segments
        assert len(results) == 4

    def test_from_token_ids(self):
        """build_round_prompt_from_token_ids must produce the same result
        as going through text."""
        tokenizer = self._get_tokenizer()
        builder = RoundAwarePromptBuilder(tokenizer, sep_str=" # # ")

        history_ids = [10, 20, 30]
        block_ids = [[40, 50], [60, 70, 80]]

        prompt = builder.build_round_prompt_from_token_ids(
            private_history_ids=history_ids,
            shared_block_ids=block_ids,
        )
        sep_ids = builder.get_separator_tokens()

        # Verify structure: history | sep | block0 | sep | block1
        expected = history_ids + sep_ids + [40, 50] + sep_ids + [60, 70, 80]
        assert prompt.tolist() == expected
