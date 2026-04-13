# TokenDance — Deployment Runbook

## Quick-Start: Running TokenDance on Your Rig

### 1. Clone & Install

```bash
# Clone the TokenDance branch
git clone -b tokendance-impl https://github.com/CuriousCaliBoi/LMCache.git
cd LMCache

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch first (match your CUDA version)
pip install torch

# Install LMCache from the tokendance-impl branch
pip install -e . --no-build-isolation

# Install vLLM
pip install vllm
```

### 2. Launch vLLM + LMCache with TokenDance Enabled

**Terminal 1 — Start the server:**

```bash
# TokenDance environment variables
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_ENABLE_BLENDING=True
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=10

# TokenDance-specific: enable segment indexing via the separator
export LMCACHE_BLEND_SPECIAL_STR="<TTSEP>"
export LMCACHE_BLEND_CHECK_LAYERS="1"
export LMCACHE_BLEND_RECOMPUTE_RATIOS="0.15"

# To use the full RoundAwareSegmentDatabase instead of the legacy
# SegmentTokenDatabase, set this:
export LMCACHE_EXTRA_CONFIG='{"enable_tokendance": true}'

# Launch vLLM with LMCache KV connector
# Replace the model with whatever fits your GPU
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --enable-prefix-caching=false \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 3. Send Multi-Agent Prompts with `<TTSEP>` Separators

**Terminal 2 — Python client using the prompt builder:**

```python
from openai import OpenAI
from transformers import AutoTokenizer

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# The separator string MUST match LMCACHE_BLEND_SPECIAL_STR
SEP = "<TTSEP>"

# ---------- Simulate a 3-agent All-Gather round ----------

# Shared output blocks from the previous round (identical across agents)
shared_block_1 = "Agent Alpha reported: The weather is sunny today with a high of 75°F."
shared_block_2 = "Agent Beta reported: Stock market indices rose 1.2% across all sectors."
shared_block_3 = "Agent Gamma reported: New policy changes were announced at 3pm EST."

# Agent-specific private histories (different per agent)
agent_prompts = {
    "Agent_A": f"You are Agent Alpha, a weather analyst.{SEP}{shared_block_1}{SEP}{shared_block_2}{SEP}{shared_block_3}{SEP}Based on all reports, what is your weather forecast?",
    "Agent_B": f"You are Agent Beta, a financial analyst.{SEP}{shared_block_1}{SEP}{shared_block_2}{SEP}{shared_block_3}{SEP}Based on all reports, what are your market predictions?",
    "Agent_C": f"You are Agent Gamma, a policy analyst.{SEP}{shared_block_1}{SEP}{shared_block_2}{SEP}{shared_block_3}{SEP}Based on all reports, what policy impacts do you foresee?",
}

# Send all agent requests
for agent_name, prompt in agent_prompts.items():
    print(f"\n{'='*60}")
    print(f"Sending request for {agent_name}...")
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
    )
    print(f"{agent_name} response: {response.choices[0].message.content[:200]}")
```

### 4. Using the Python Prompt Builder (Recommended)

For tighter integration, use the `RoundAwarePromptBuilder` directly:

```python
from transformers import AutoTokenizer
from lmcache.v1.tokendance import RoundAwarePromptBuilder

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
builder = RoundAwarePromptBuilder(tokenizer, sep_str="<TTSEP>")

# Build token-level prompts with proper separator injection
prompt_tokens = builder.build_round_prompt(
    private_history_text="You are Agent Alpha, a weather analyst.",
    shared_blocks=[
        "Agent Alpha reported: The weather is sunny today.",
        "Agent Beta reported: Stock market rose 1.2%.",
        "Agent Gamma reported: New policy changes announced.",
    ],
    round_task_text="Based on all reports, what is your forecast?",
)

# Send to vLLM via token IDs
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Convert to text for the chat API, or use the completions API with token IDs
prompt_text = tokenizer.decode(prompt_tokens)
response = client.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    prompt=prompt_text,
    max_tokens=200,
)
print(response.choices[0].text)
```

### 5. Run the Benchmark

```bash
cd LMCache
python benchmarks/tokendance_benchmark.py
```

### 6. Verify It's Working

Look for these log lines in the vLLM server output:

```
LMCache INFO: Using RoundAwareSegmentDatabase (TokenDance)
LMCache INFO: RoundAwareSegmentDatabase: sep_str='<TTSEP>'  sep_ids=[...]  sep_len=...
LMCache INFO: SegmentHashTable created (initial_capacity=4096)
```

When the second and third agents hit the same shared blocks, you should see
cache hits in the LMCache logs — the shared blocks produce the same content
hash regardless of position.

### 7. Config Reference

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `LMCACHE_EXTRA_CONFIG='{"enable_tokendance":true}'` | Use `RoundAwareSegmentDatabase` | `false` |
| `LMCACHE_BLEND_SPECIAL_STR` | Separator string | `" # # "` |
| `LMCACHE_ENABLE_BLENDING` | Enable blending/reuse | `false` |
| `LMCACHE_USE_LAYERWISE` | Layerwise KV transfer | `false` |
| `LMCACHE_BLEND_CHECK_LAYERS` | Layers for diff check | `1` |
| `LMCACHE_BLEND_RECOMPUTE_RATIOS` | Fraction to recompute | `0.15` |
| `LMCACHE_CHUNK_SIZE` | Tokens per chunk | `256` |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | CPU cache pool (GB) | `5.0` |

### Troubleshooting

- **"CUDA_HOME not set"**: Install CUDA toolkit or set `export CUDA_HOME=/usr/local/cuda`
- **OOM on large models**: Reduce `--gpu-memory-utilization` or use a smaller model
- **No cache hits**: Verify `LMCACHE_BLEND_SPECIAL_STR` matches the separator in your prompts exactly
- **Fallback to ChunkedTokenDatabase**: Check that `LMCACHE_EXTRA_CONFIG` includes `"enable_tokendance": true`
