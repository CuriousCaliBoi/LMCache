# SPDX-License-Identifier: Apache-2.0
"""
Tests for TokenDance Components 2–4.

- Component 2: Collective KV Cache Reuse (KVCollector, find_compatible_groups)
- Component 3: Diff-Aware Storage (BlockSparseDiff, MirrorObject, DiffAwareStorageWrapper)
- Component 4: Fused Sparse Restore (FusedDiffRestorer, diff kernels)
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
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
    compute_block_sparse_diff,
)
from lmcache.v1.tokendance.fused_restore import (
    FusedDiffRestorer,
    apply_kv_diff_kernel,
    apply_paired_kv_diff_kernel,
    rope_recover,
)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _make_cache_key(chunk_hash: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
    )


def _make_agent_request(
    req_id: str,
    round_id: str = "round_0",
    prompt_len: int = 100,
    cached_span: int = 50,
    slot_start: int = 0,
    segment_hashes: list = None,
) -> AgentRequest:
    return AgentRequest(
        req_id=req_id,
        round_id=round_id,
        prompt_tokens=list(range(prompt_len)),
        prompt_len=prompt_len,
        cached_span=cached_span,
        slot_mapping=list(range(slot_start, slot_start + prompt_len)),
        segment_hashes=segment_hashes or [1000, 2000, 3000],
    )


# ───────────────────────────────────────────────────────────────────────────
# Component 2: Collective KV Cache Reuse
# ───────────────────────────────────────────────────────────────────────────


class TestFindCompatibleGroups:
    """Tests for the compatibility grouping function."""

    def test_same_round_same_shape(self):
        """Requests with matching round/len/cached_span and non-overlapping
        slots go in one group."""
        reqs = [
            _make_agent_request("a", slot_start=0),
            _make_agent_request("b", slot_start=100),
            _make_agent_request("c", slot_start=200),
        ]
        groups = find_compatible_groups(reqs)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_different_rounds_split(self):
        """Requests from different rounds go in separate groups."""
        reqs = [
            _make_agent_request("a", round_id="r1", slot_start=0),
            _make_agent_request("b", round_id="r2", slot_start=0),
        ]
        groups = find_compatible_groups(reqs)
        assert len(groups) == 2

    def test_different_prompt_len_split(self):
        """Requests with different prompt lengths go in separate groups."""
        reqs = [
            _make_agent_request("a", prompt_len=100, slot_start=0),
            _make_agent_request("b", prompt_len=200, slot_start=0),
        ]
        groups = find_compatible_groups(reqs)
        assert len(groups) == 2

    def test_overlapping_slots_split(self):
        """Overlapping slot mappings cause a new sub-group."""
        reqs = [
            _make_agent_request("a", slot_start=0),   # slots 0..99
            _make_agent_request("b", slot_start=50),   # slots 50..149 (overlap)
        ]
        groups = find_compatible_groups(reqs)
        assert len(groups) == 2

    def test_empty_input(self):
        """No requests → no groups."""
        groups = find_compatible_groups([])
        assert groups == []


class TestKVCollector:
    """Tests for the collective reuse orchestrator."""

    def test_shared_segment_detection(self):
        """begin_group should identify segments common to all requests."""
        collector = KVCollector(num_layers=2)
        reqs = [
            _make_agent_request("a", segment_hashes=[10, 20, 30]),
            _make_agent_request("b", segment_hashes=[20, 30, 40]),
            _make_agent_request("c", segment_hashes=[20, 30, 50]),
        ]
        collector.begin_group(reqs)
        assert set(collector._shared_hashes) == {20, 30}

    def test_process_layer_selects_important_positions(self):
        """On a check layer, process_layer should identify important positions
        and only refresh those."""
        collector = KVCollector(
            num_layers=2, check_layers=[0], recompute_ratio=0.5
        )
        reqs = [_make_agent_request("a"), _make_agent_request("b", slot_start=100)]
        collector.begin_group(reqs)

        # Create fake tensors
        hidden = 8
        n_tok = 10
        q = {"a": torch.randn(n_tok, hidden), "b": torch.randn(n_tok, hidden)}
        k = {"a": torch.randn(n_tok, hidden), "b": torch.randn(n_tok, hidden)}
        v = {"a": torch.randn(n_tok, hidden), "b": torch.randn(n_tok, hidden)}
        # Cached K differs from fresh K at some positions
        cached_k = {"a": k["a"] + 0.01, "b": k["b"] + 0.01}
        cached_k["a"][3] += 10.0  # make position 3 important for "a"
        cached_k["b"][7] += 10.0  # make position 7 important for "b"
        cached_v = {"a": torch.randn(n_tok, hidden), "b": torch.randn(n_tok, hidden)}

        updated = collector.process_layer(
            layer_id=0,
            q_tensors=q,
            k_tensors=k,
            v_tensors=v,
            cached_k=cached_k,
            cached_v=cached_v,
        )

        # Should have updated results for both requests
        assert "a" in updated and "b" in updated
        # Important positions should have been recorded
        assert 3 in collector._imp_positions.get("a", [])
        assert 7 in collector._imp_positions.get("b", [])

    def test_finish_group_selects_master(self):
        """finish_group should produce a ReusePlan with the lowest-deviation
        request as master."""
        collector = KVCollector(num_layers=1, check_layers=[0])
        reqs = [_make_agent_request("a"), _make_agent_request("b", slot_start=100)]
        collector.begin_group(reqs)

        hidden = 4
        n_tok = 4
        k_a = torch.zeros(n_tok, hidden)
        k_b = torch.zeros(n_tok, hidden)
        cached_k_a = torch.ones(n_tok, hidden) * 0.01  # small deviation
        cached_k_b = torch.ones(n_tok, hidden) * 10.0  # large deviation

        collector.process_layer(
            layer_id=0,
            q_tensors={"a": torch.randn(n_tok, hidden), "b": torch.randn(n_tok, hidden)},
            k_tensors={"a": k_a, "b": k_b},
            v_tensors={"a": torch.randn(n_tok, hidden), "b": torch.randn(n_tok, hidden)},
            cached_k={"a": cached_k_a, "b": cached_k_b},
            cached_v={"a": torch.randn(n_tok, hidden), "b": torch.randn(n_tok, hidden)},
        )

        plan = collector.finish_group()
        assert isinstance(plan, ReusePlan)
        # "a" should be master because its deviation is smaller
        assert plan.master_req_id == "a"
        assert set(plan.group_req_ids) == {"a", "b"}


# ───────────────────────────────────────────────────────────────────────────
# Component 3: Diff-Aware Storage
# ───────────────────────────────────────────────────────────────────────────


class TestBlockSparseDiff:
    """Tests for the block-sparse diff representation."""

    def test_compute_identical_tensors(self):
        """Identical tensors should yield an empty diff."""
        k = torch.randn(64, 16)
        diff = compute_block_sparse_diff(k, k.clone(), k, k.clone(), block_size=32)
        assert diff.is_empty()
        assert diff.compression_ratio == float("inf")

    def test_compute_different_tensors(self):
        """Completely different tensors should mark all blocks."""
        master_k = torch.zeros(64, 16)
        master_v = torch.zeros(64, 16)
        mirror_k = torch.ones(64, 16)
        mirror_v = torch.ones(64, 16)
        diff = compute_block_sparse_diff(
            master_k, master_v, mirror_k, mirror_v, block_size=32
        )
        assert diff.num_diff_blocks == 2  # 64/32 = 2 blocks
        assert diff.k_corrections is not None
        assert diff.v_corrections is not None
        assert diff.shared_indices is True

    def test_partial_diff(self):
        """Only blocks that differ should be recorded."""
        master_k = torch.zeros(96, 8)
        master_v = torch.zeros(96, 8)
        mirror_k = torch.zeros(96, 8)
        mirror_v = torch.zeros(96, 8)
        # Change only the second block (tokens 32..63)
        mirror_k[32:64] = 1.0
        mirror_v[32:64] = 1.0
        diff = compute_block_sparse_diff(
            master_k, master_v, mirror_k, mirror_v, block_size=32
        )
        assert diff.num_diff_blocks == 1
        assert diff.block_indices == [1]
        assert diff.compression_ratio == 3.0  # 3 total blocks / 1 diff block


class TestMirrorObject:
    """Tests for the Mirror proxy object."""

    def test_is_mirror(self):
        """MirrorObject.is_mirror should always be True."""
        m = MirrorObject(
            master_key=_make_cache_key(1),
            mirror_key=_make_cache_key(2),
            diffs={},
        )
        assert m.is_mirror is True

    def test_footprint_ratio_empty(self):
        """Empty diffs → zero footprint ratio."""
        m = MirrorObject(
            master_key=_make_cache_key(1),
            mirror_key=_make_cache_key(2),
            diffs={},
        )
        assert m.memory_footprint_ratio() == 0.0


class TestDiffAwareStorageWrapper:
    """Tests for the storage wrapper."""

    def test_master_mirror_lifecycle(self):
        """Register a master, store a mirror, retrieve it."""
        wrapper = DiffAwareStorageWrapper(block_size=32)
        master_key = _make_cache_key(100)
        mirror_key = _make_cache_key(200)

        wrapper.register_master(master_key)
        assert wrapper.is_master(master_key)
        assert not wrapper.is_mirror(master_key)

        mirror = MirrorObject(
            master_key=master_key,
            mirror_key=mirror_key,
            diffs={0: BlockSparseDiff(num_tokens=64, block_size=32, block_indices=[1])},
            num_tokens=64,
        )
        wrapper.store_mirror(mirror)
        assert wrapper.is_mirror(mirror_key)

        retrieved = wrapper.get(mirror_key)
        assert retrieved is not None
        assert retrieved.master_key == master_key

    def test_remove(self):
        """remove() should clear both master and mirror registrations."""
        wrapper = DiffAwareStorageWrapper()
        key = _make_cache_key(300)
        wrapper.register_master(key)
        assert wrapper.is_master(key)
        wrapper.remove(key)
        assert not wrapper.is_master(key)

    def test_stats(self):
        """get_stats should reflect current state."""
        wrapper = DiffAwareStorageWrapper()
        wrapper.register_master(_make_cache_key(1))
        wrapper.store_mirror(MirrorObject(
            master_key=_make_cache_key(1),
            mirror_key=_make_cache_key(2),
            diffs={},
        ))
        stats = wrapper.get_stats()
        assert stats["num_masters"] == 1
        assert stats["num_mirrors"] == 1

    def test_token_similarity_fallback(self):
        """build_mirror_by_token_similarity should produce a valid MirrorObject."""
        master_k = torch.zeros(64, 8)
        master_v = torch.zeros(64, 8)
        mirror_k = torch.zeros(64, 8)
        mirror_v = torch.zeros(64, 8)
        mirror_k[32:64] = 1.0
        mirror = build_mirror_by_token_similarity(
            master_k, master_v, mirror_k, mirror_v,
            _make_cache_key(1), _make_cache_key(2),
            block_size=32,
        )
        assert mirror.is_mirror
        assert 0 in mirror.diffs


# ───────────────────────────────────────────────────────────────────────────
# Component 4: Fused Sparse Restore
# ───────────────────────────────────────────────────────────────────────────


class TestDiffKernels:
    """Tests for the Python-reference diff kernels."""

    def test_single_plane_kernel(self):
        """apply_kv_diff_kernel should add corrections at the right positions."""
        buf = torch.zeros(64, 8)
        corrections = torch.ones(1, 32, 8)  # one block
        apply_kv_diff_kernel(buf, [1], corrections, block_size=32)
        # Block 1 = tokens 32..63 should be 1.0
        assert torch.allclose(buf[32:64], torch.ones(32, 8))
        # Block 0 should be unchanged
        assert torch.allclose(buf[0:32], torch.zeros(32, 8))

    def test_paired_kernel(self):
        """apply_paired_kv_diff_kernel should correct both K and V."""
        k_buf = torch.zeros(32, 4)
        v_buf = torch.zeros(32, 4)
        k_corr = torch.ones(1, 32, 4) * 2.0
        v_corr = torch.ones(1, 32, 4) * 3.0
        apply_paired_kv_diff_kernel(
            k_buf, v_buf, [0], k_corr, v_corr, block_size=32
        )
        assert torch.allclose(k_buf, torch.ones(32, 4) * 2.0)
        assert torch.allclose(v_buf, torch.ones(32, 4) * 3.0)

    def test_multiple_blocks(self):
        """Corrections across multiple non-contiguous blocks."""
        buf = torch.zeros(128, 4)
        corrections = torch.ones(2, 32, 4)
        apply_kv_diff_kernel(buf, [0, 3], corrections, block_size=32)
        assert torch.allclose(buf[0:32], torch.ones(32, 4))
        assert torch.allclose(buf[32:64], torch.zeros(32, 4))
        assert torch.allclose(buf[64:96], torch.zeros(32, 4))
        assert torch.allclose(buf[96:128], torch.ones(32, 4))

    def test_partial_last_block(self):
        """Corrections on a partial last block should not overflow."""
        buf = torch.zeros(50, 4)  # not a multiple of 32
        corrections = torch.ones(1, 32, 4)
        apply_kv_diff_kernel(buf, [1], corrections, block_size=32)
        # Block 1 = tokens 32..49 (only 18 tokens)
        assert torch.allclose(buf[32:50], torch.ones(18, 4))


class TestFusedDiffRestorer:
    """Tests for the fused layerwise restore."""

    def test_restore_with_diff(self):
        """Restoring a mirror should produce master + corrections."""
        hidden = 8
        n_tok = 64
        master_k = torch.zeros(n_tok, hidden)
        master_v = torch.zeros(n_tok, hidden)
        mirror_k = torch.zeros(n_tok, hidden)
        mirror_v = torch.zeros(n_tok, hidden)
        mirror_k[32:64] = 5.0
        mirror_v[32:64] = 7.0

        diff = compute_block_sparse_diff(
            master_k, master_v, mirror_k, mirror_v, block_size=32, layer_id=0
        )
        mirror = MirrorObject(
            master_key=_make_cache_key(1),
            mirror_key=_make_cache_key(2),
            diffs={0: diff},
            num_tokens=n_tok,
        )

        def master_loader(layer_id):
            return master_k.clone(), master_v.clone()

        restorer = FusedDiffRestorer(num_layers=1)
        result = restorer.restore(mirror, master_loader)

        assert 0 in result
        k_restored, v_restored = result[0]
        assert torch.allclose(k_restored[0:32], torch.zeros(32, hidden))
        assert torch.allclose(k_restored[32:64], torch.ones(32, hidden) * 5.0)
        assert torch.allclose(v_restored[32:64], torch.ones(32, hidden) * 7.0)

    def test_restore_empty_diff(self):
        """Restoring a mirror with no diff should return master data."""
        hidden = 4
        n_tok = 32
        master_k = torch.randn(n_tok, hidden)
        master_v = torch.randn(n_tok, hidden)

        mirror = MirrorObject(
            master_key=_make_cache_key(1),
            mirror_key=_make_cache_key(2),
            diffs={0: BlockSparseDiff(num_tokens=n_tok, block_size=32, block_indices=[])},
            num_tokens=n_tok,
        )

        def master_loader(layer_id):
            return master_k.clone(), master_v.clone()

        restorer = FusedDiffRestorer(num_layers=1)
        result = restorer.restore(mirror, master_loader)
        k_r, v_r = result[0]
        assert torch.allclose(k_r, master_k)
        assert torch.allclose(v_r, master_v)

    def test_dense_fallback(self):
        """Dense fallback should return unmodified master data."""
        hidden = 4
        master_k = torch.randn(16, hidden)
        master_v = torch.randn(16, hidden)

        def master_loader(layer_id):
            return master_k.clone(), master_v.clone()

        restorer = FusedDiffRestorer(num_layers=2)
        result = restorer.restore_dense_fallback(master_loader)
        assert len(result) == 2
        for layer_id in range(2):
            k_r, v_r = result[layer_id]
            assert torch.allclose(k_r, master_k)

    def test_multi_layer_restore(self):
        """Restore across multiple layers with per-layer diffs."""
        hidden = 4
        n_tok = 64

        diffs = {}
        for layer_id in range(3):
            mk = torch.zeros(n_tok, hidden)
            mv = torch.zeros(n_tok, hidden)
            rk = torch.zeros(n_tok, hidden)
            rv = torch.zeros(n_tok, hidden)
            rk[layer_id * 16 : (layer_id + 1) * 16] = float(layer_id + 1)
            rv[layer_id * 16 : (layer_id + 1) * 16] = float(layer_id + 1)
            diffs[layer_id] = compute_block_sparse_diff(
                mk, mv, rk, rv, block_size=16, layer_id=layer_id
            )

        mirror = MirrorObject(
            master_key=_make_cache_key(1),
            mirror_key=_make_cache_key(2),
            diffs=diffs,
            num_tokens=n_tok,
        )

        def master_loader(layer_id):
            return torch.zeros(n_tok, hidden), torch.zeros(n_tok, hidden)

        restorer = FusedDiffRestorer(num_layers=3)
        result = restorer.restore(mirror, master_loader)

        for layer_id in range(3):
            k_r, v_r = result[layer_id]
            expected = torch.zeros(n_tok, hidden)
            expected[layer_id * 16 : (layer_id + 1) * 16] = float(layer_id + 1)
            assert torch.allclose(k_r, expected), f"K mismatch at layer {layer_id}"
            assert torch.allclose(v_r, expected), f"V mismatch at layer {layer_id}"


class TestRopeRecover:
    """Tests for RoPE position recovery."""

    def test_no_rope_fn_is_noop(self):
        """Without a rope_fn, keys should pass through unchanged."""
        k = torch.randn(10, 8)
        old_pos = torch.arange(10)
        new_pos = torch.arange(10, 20)
        result = rope_recover(old_pos, new_pos, k, rope_fn=None)
        assert torch.equal(result, k)

    def test_same_positions_is_noop(self):
        """When old == new positions, keys should pass through unchanged."""
        k = torch.randn(10, 8)
        pos = torch.arange(10)
        result = rope_recover(pos, pos, k, rope_fn=lambda p, q, k: (q, k))
        assert torch.equal(result, k)
