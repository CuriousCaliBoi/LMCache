# TokenDance — Agent Deployment & Benchmarking Instructions

## Context: What Is This?

You are working with the `tokendance-impl` branch of an LMCache fork at:
```
https://github.com/CuriousCaliBoi/LMCache
```

This branch implements **TokenDance**, a research system from the paper "TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing" (arXiv:2604.03143). It augments LMCache + vLLM to make multi-agent LLM workloads dramatically more efficient by exploiting the fact that agents in the same round share most of their prompt content.

The implementation lives in `lmcache/v1/tokendance/` and consists of four components:

1. **Round-Aware Segment Indexing** (`segment_index.py`, `prompt_interface.py`) — Splits prompts at `<TTSEP>` separator tokens and hashes each segment by *content only* (position-independent). Two agents sharing the same output block get the same cache key regardless of where that block appears in their prompt.

2. **Collective KV Cache Reuse** (`collective_reuse.py`) — Groups compatible agent requests from the same round, performs batched RoPE rotation and important-position selection *once* for the group instead of N times.

3. **Diff-Aware Storage** (`diff_storage.py`) — Stores one dense "Master" KV cache and encodes each remaining agent's cache as a block-sparse diff (10-20% of full size). The paper shows 11-17× compression.

4. **Fused Sparse Restore** (`fused_restore.py`, `csrc/tokendance/fused_restore_kernels.cu`) — Applies sparse corrections during the GPU transfer pipeline in a ping-pong fashion without ever materializing a separate dense copy.

All 48 unit tests pass. The CUDA kernels have Python-reference fallbacks for CPU-only testing.

---

## What You Need To Do

### Phase 1: Setup & Verify

```bash
# 1. Clone the tokendance-impl branch
git clone -b tokendance-impl https://github.com/CuriousCaliBoi/LMCache.git
cd LMCache

# 2. Create environment
python -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch (match your CUDA version — e.g. CUDA 12.x)
pip install torch

# 4. Install LMCache with CUDA extensions
pip install -e . --no-build-isolation

# 5. Install vLLM
pip install vllm

# 6. Run the unit tests to verify everything works
pip install pytest
pytest tests/v1/test_tokendance_segment_index.py tests/v1/test_tokendance_components.py -v
# Expected: 48 passed
```

### Phase 2: Launch vLLM + TokenDance Server

Pick a model that fits your GPU. Qwen2.5-7B-Instruct works on a single A100/H100/4090. For 14B use a larger GPU or reduce context length.

```bash
# Terminal 1: Launch the server
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_ENABLE_BLENDING=True \
LMCACHE_USE_LAYERWISE=True \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=10 \
LMCACHE_BLEND_SPECIAL_STR="<TTSEP>" \
LMCACHE_BLEND_CHECK_LAYERS="1" \
LMCACHE_BLEND_RECOMPUTE_RATIOS="0.15" \
LMCACHE_EXTRA_CONFIG='{"enable_tokendance": true}' \
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

**Confirm it started correctly** — look for these log lines in the server output:
```
LMCache INFO: Using RoundAwareSegmentDatabase (TokenDance)
LMCache INFO: SegmentHashTable created
```

If you instead see `ChunkedTokenDatabase`, the `enable_tokendance` flag is not being picked up — double-check the `LMCACHE_EXTRA_CONFIG` env var.

### Phase 3: Benchmark — Baseline vs TokenDance

You need to run two server configurations and compare them. Create this benchmark script:

```python
#!/usr/bin/env python3
"""benchmark_tokendance_e2e.py — End-to-end multi-agent benchmark."""
import time
import statistics
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def make_agent_prompt(agent_id: int, num_shared_blocks: int, block_size: int, sep: str) -> str:
    """Build one agent's prompt with private history + shared blocks."""
    # Private history (unique per agent)
    private = f"You are Agent {agent_id}. Your role is to analyze reports from all other agents and provide your unique perspective based on your expertise area {agent_id}. " * 20

    # Shared blocks (identical across agents — this is the All-Gather pattern)
    shared_blocks = []
    for b in range(num_shared_blocks):
        block_text = f"Shared report block {b}: " + f"This is analysis content that is shared across all agents in round. It contains detailed information about topic {b}. " * block_size
        shared_blocks.append(block_text)

    # Round task (unique per agent)
    task = f"Based on all the shared reports above, provide your Agent {agent_id} analysis. Be concise."

    # Assemble with separators
    parts = [private]
    for block in shared_blocks:
        parts.append(sep + block)
    parts.append(sep + task)
    return "".join(parts)


def run_round(client, model, num_agents, num_shared_blocks, block_size, sep, max_tokens=50):
    """Send one All-Gather round of agent requests and measure timing."""
    prompts = [
        make_agent_prompt(i, num_shared_blocks, block_size, sep)
        for i in range(num_agents)
    ]

    ttfts = []
    latencies = []

    def send_one(prompt):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        t1 = time.perf_counter()
        return t1 - t0

    # Send sequentially to measure individual TTFT impact
    for prompt in prompts:
        latency = send_one(prompt)
        latencies.append(latency)

    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-agents", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--num-shared-blocks", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=10, help="Repetitions per block (controls length)")
    parser.add_argument("--sep", type=str, default="<TTSEP>", help="Separator string")
    parser.add_argument("--label", type=str, default="unknown", help="System label for output")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")

    # Warmup
    print("Warming up...")
    client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5,
    )

    print(f"\n{'='*70}")
    print(f"System: {args.label}")
    print(f"{'='*70}")
    print(f"{'Agents':>8} | {'Avg Latency (ms)':>18} | {'P50 (ms)':>10} | {'P99 (ms)':>10} | {'Throughput (req/s)':>20}")
    print("-" * 70)

    for num_agents in args.num_agents:
        all_latencies = []
        for round_id in range(args.num_rounds):
            round_lats = run_round(
                client, args.model, num_agents,
                args.num_shared_blocks, args.block_size, args.sep
            )
            all_latencies.extend(round_lats)

        avg_ms = statistics.mean(all_latencies) * 1000
        p50_ms = statistics.median(all_latencies) * 1000
        p99_ms = sorted(all_latencies)[int(len(all_latencies) * 0.99)] * 1000
        total_time = sum(all_latencies)
        throughput = len(all_latencies) / total_time

        print(f"{num_agents:>8} | {avg_ms:>18.1f} | {p50_ms:>10.1f} | {p99_ms:>10.1f} | {throughput:>20.1f}")

    print()


if __name__ == "__main__":
    main()
```

### Run the benchmark in two configurations:

**A) Baseline — vLLM with prefix caching, no TokenDance:**

```bash
# Terminal 1: Launch baseline server (no LMCache, just vLLM prefix caching)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --enable-prefix-caching

# Terminal 2: Run benchmark
python benchmark_tokendance_e2e.py \
    --port 8000 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-agents 2 4 8 \
    --num-rounds 3 \
    --sep "<TTSEP>" \
    --label "Baseline (vLLM + Prefix Caching)"
```

**B) TokenDance — LMCache with round-aware segment indexing:**

```bash
# Terminal 1: Launch TokenDance server (kill the baseline first)
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_ENABLE_BLENDING=True \
LMCACHE_USE_LAYERWISE=True \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=10 \
LMCACHE_BLEND_SPECIAL_STR="<TTSEP>" \
LMCACHE_BLEND_CHECK_LAYERS="1" \
LMCACHE_BLEND_RECOMPUTE_RATIOS="0.15" \
LMCACHE_EXTRA_CONFIG='{"enable_tokendance": true}' \
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# Terminal 2: Run the same benchmark
python benchmark_tokendance_e2e.py \
    --port 8000 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-agents 2 4 8 \
    --num-rounds 3 \
    --sep "<TTSEP>" \
    --label "TokenDance"
```

### Phase 4: What To Measure & Report

Produce a results table comparing the two configurations:

| Metric | Baseline | TokenDance | Improvement |
|--------|----------|------------|-------------|
| Avg latency (ms) at 2 agents | ? | ? | ?× |
| Avg latency (ms) at 4 agents | ? | ? | ?× |
| Avg latency (ms) at 8 agents | ? | ? | ?× |
| P99 latency (ms) at 8 agents | ? | ? | ?× |
| Throughput (req/s) at 8 agents | ? | ? | ?× |
| KV cache hit rate | ~prefix only | ~85% shared segments | — |

Also monitor GPU memory usage during the runs:
```bash
# In a third terminal, while the benchmark runs:
watch -n 1 nvidia-smi
```

Report peak GPU memory for each configuration at each agent count.

### Expected Results (from the paper)

On representative All-Gather workloads the paper reports:
- **2.3×** end-to-end latency reduction vs vLLM prefix caching
- **1.9×** prefill speedup vs per-request PIC (CacheBlend)
- **11-17×** KV cache compression (Master-Mirror storage)
- **94%** KV cache storage reduction
- **2.7×** more concurrent agents under the same latency SLO

Your numbers will depend on model size, GPU, prompt length, and sharing ratio. The gains are most visible at higher agent counts (4+) where cross-agent redundancy is highest.

---

## File Map

```
lmcache/v1/tokendance/
├── __init__.py              # Package exports (18 public symbols)
├── prompt_interface.py      # RoundAwarePromptBuilder — app inserts <TTSEP>
├── segment_index.py         # SegmentHashTable + RoundAwareSegmentDatabase
├── collective_reuse.py      # KVCollector + find_compatible_groups + ReusePlan
├── diff_storage.py          # BlockSparseDiff + MirrorObject + DiffAwareStorageWrapper
└── fused_restore.py         # FusedDiffRestorer + diff kernel Python references

csrc/tokendance/
└── fused_restore_kernels.cu # CUDA kernels: single-plane + paired K/V diff

tests/v1/
├── test_tokendance_segment_index.py  # 21 tests for Components 1
└── test_tokendance_components.py     # 27 tests for Components 2-4

benchmarks/
└── tokendance_benchmark.py  # Synthetic CPU benchmark (runs without GPU)

docs/
└── tokendance_runbook.md    # Deployment guide
```

## Troubleshooting

- **Server crashes on startup**: Check CUDA/PyTorch version compatibility. Run `python -c "import torch; print(torch.cuda.is_available())"` first.
- **"enable_tokendance not recognized"**: The config is passed via `LMCACHE_EXTRA_CONFIG` as JSON — ensure the quotes and escaping are correct for your shell.
- **No speedup visible**: Make sure the shared blocks in your prompts are long enough (500+ tokens each) and that you're using the `<TTSEP>` separator that matches `LMCACHE_BLEND_SPECIAL_STR`.
- **Import errors**: Make sure you installed from the `tokendance-impl` branch, not `dev` or `main`.
