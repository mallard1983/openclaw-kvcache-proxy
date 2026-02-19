# OpenClaw → llama-server KV Cache Proxy

A lightweight FastAPI proxy that sits between [OpenClaw](https://github.com/openclaw/openclaw) and [llama-server](https://github.com/ggml-org/llama.cpp) (llama.cpp) to dramatically improve KV cache hit rates and reduce per-turn latency.

## Test Hardware

Developed and benchmarked on a **Minisforum Ms-S1 Max** (AMD Strix Halo / Halo Strix platform), running Windows 11 Pro with llama.cpp's Vulkan backend.

| Component | Detail |
|-----------|--------|
| Device | Minisforum Ms-S1 Max |
| Platform | AMD Strix Halo (Halo Strix) |
| GPU | AMD Radeon 8060S |
| Memory | ~128 GB unified (UMA) — ~106 GB available to Vulkan |
| Backend | llama.cpp Vulkan (`ggml_vulkan`) |
| OS | Windows 11 Pro |
| CPU threads | 16 inference / 32 total |
| Model | Qwen3-Coder-Next 80B Q6_K (61 GiB, fully GPU-offloaded) |
| llama-server flags | `--cache-prompt --cont-batching --parallel 4 -ngl 99 --ctx-size 131072` |

The Strix Halo platform is particularly well-suited for large local models: the 8060S's unified memory pool lets llama.cpp fully offload a 61 GiB model (49/49 layers) to the GPU with no CPU fallback, while the large RAM headroom keeps KV cache and recurrent state buffers well clear of any pressure. If you're on a similar AMD Strix Halo mini-PC or laptop, the numbers in this repo should be directly comparable.

## The Problem

OpenClaw injects two volatile fields into every request that bust llama-server's prompt cache on every single turn, even when the conversation hasn't meaningfully changed:

1. **`"message_id": "<UUID>"`** in the system prompt's Inbound Context section
   This UUID is unique per-message and appears roughly 2,500 tokens into the ~50KB system prompt. Everything after it — 85% of the prompt — is seen as "new" by the cache every turn.

2. **`[Wed 2026-02-18 20:48 UTC]`** timestamp prefixes in each user message
   Injected per-message, these bust the conversation history prefix too.

The result without this proxy: llama-server reports `sim_best ≈ 0.15` (15% cache similarity) and forces full prompt re-processing on every turn — even for a one-word reply. On a 16K-token prompt, that's **~38–40 seconds of prompt eval time per turn** with no benefit.

## What This Proxy Does

Before forwarding each request to llama-server, the proxy strips the volatile fields:

- Removes `"message_id": "..."` from all JSON metadata blocks (system prompt + per-message wrappers)
- Removes `[Day YYYY-MM-DD HH:MM UTC]` timestamp prefixes from user message text

The conversation content is untouched. The model sees identical output — only the cache-busting metadata is gone.

## Results

Tested with **Qwen3-Coder-Next (80B, Q6_K)** on llama-server with `--cache-prompt`:

| Metric | Without proxy | With proxy |
|--------|--------------|------------|
| `sim_best` (cache similarity) | 0.151 every turn | **0.943–1.000** |
| Prompt re-processing | Full 16K+ tokens every turn | Only new tokens since last checkpoint |
| Turn 2 prompt eval time | ~39,000 ms | **~1,700 ms** |
| "forcing full prompt re-processing" | Every turn | Gone |
| Generation speed | ~33 tok/s | ~30–33 tok/s (context-length dependent) |

The ~22× speedup on prompt evaluation is the headline number. Generation speed (output tokens) is hardware-bound and unaffected.

> **Note on Qwen3-Coder-Next / hybrid SSM architectures:** This model uses a hybrid attention + recurrent (SSM/Mamba) architecture. Older llama.cpp builds forced full re-processing due to SWA/recurrent state limitations regardless of cache similarity. With a stable prefix (sim_best ≈ 1.0), the checkpoint system works correctly and context is properly restored between turns.

## Setup

### Requirements

```
pip install fastapi uvicorn httpx
```

Python 3.10+ required (uses `match`-friendly type hints).

### Configuration

Edit the top of `proxy.py`:

```python
LISTEN_PORT = 1234        # Port OpenClaw connects to
BACKEND_URL = "http://localhost:12345"  # Your llama-server address

STRIP_MESSAGE_IDS = True  # Primary fix — removes message_id UUIDs
STRIP_TIMESTAMPS  = True  # Secondary fix — removes [Day HH:MM UTC] prefixes
```

### Running

```bash
python proxy.py
```

Point OpenClaw at `http://<your-host>:1234` instead of directly at llama-server. Everything else — model names, tool definitions, streaming — passes through unchanged.

### llama-server flags

Make sure llama-server is running with prompt caching enabled:

```
--cache-prompt       # Required — enables the prompt cache
--ctx-size 131072    # Large enough context for your use case
--parallel 4         # Slots for concurrent requests
```

## How It Works

OpenClaw's system prompt includes an **Inbound Context** section injected per-request:

```json
{
  "schema": "openclaw.inbound_meta.v1",
  "message_id": "775b2410-8917-4fad-af9d-cfcbf526eee8",  ← changes every turn
  "sender_id": "openclaw-control-ui",
  "channel": "webchat",
  ...
}
```

This sits at roughly token 2,500 in a ~16,500-token prompt. Since llama-server's LCP (Longest Common Prefix) cache matching starts from token 0, diverging at position 2,500 means only the first 15% of the prompt is reusable — the rest is re-evaluated from scratch every single turn.

By removing the `message_id` line, the system prompt becomes identical across turns. The cache now covers 99%+ of the prompt, and llama-server only processes the genuinely new content (the latest user message and any tool results).

Each user message also wraps the human text in a metadata block:

```
Conversation info (untrusted metadata):
```json
{"message_id": "e6d298e0-...", "sender": "openclaw-control-ui"}
```
[Wed 2026-02-18 20:48 UTC] Hello.
```

The proxy strips the `message_id` and the timestamp, leaving:

```
Conversation info (untrusted metadata):
```json
{"sender": "openclaw-control-ui"}
```
Hello.
```

Since the model is instructed to treat this block as untrusted metadata, this has no effect on model behavior.

## Files

| File | Purpose |
|------|---------|
| `proxy.py` | The optimization proxy — run this |
| `llm_proxy_logger.py` | Logging-only proxy for capturing and analysing your own traffic before writing normalizations |
| `replay.py` | Replays requests captured by the logger through the proxy for benchmarking |
| `inspect_log.py` | Parses a captured log to summarize request structure (item counts, tool calls, content-lengths) |

Log files (`proxy.log`, `proxy_capture.log`) and llama-server output (`llama*.txt`) are excluded from the repo via `.gitignore` — they contain local system paths and conversation content. Run the logger against your own setup to generate them.

## Logging

The proxy logs one line per request showing what it normalized:

```
POST /v1/responses | items=8 | ts_removed=2 | msg_ids_removed=3 | items_modified=3 | stream=True
```

- `items` — total items in the input array (system + conversation history)
- `ts_removed` — timestamp prefixes stripped
- `msg_ids_removed` — message_id fields stripped
- `items_modified` — how many input items were changed

## Limitations

- **Generation speed is unchanged.** The proxy eliminates the prompt re-processing penalty but doesn't affect how fast the model produces output tokens — that's hardware-bound.
- **Cold start is still slow.** The first turn of a new session always does a full prompt eval (~38s for a 16K system prompt). Every subsequent turn benefits from the cache.
- **Context growth gradually slows generation.** As the conversation grows (more tool calls, longer history), generation speed drifts down from ~33 tok/s toward ~27–28 tok/s at 26K tokens. This is normal attention scaling, not a proxy issue.
- **OpenClaw-specific.** The `message_id` patterns and timestamp format are specific to OpenClaw's request structure. Other clients may need different normalization rules.

## Background

This proxy was developed by capturing and analyzing live traffic between OpenClaw and llama-server, then identifying the specific fields responsible for cache misses. The analysis tools (`llm_proxy_logger.py`, `inspect_log.py`) are included if you want to do the same analysis for a different client.

The key diagnostic is llama-server's `sim_best` value in its slot log output. If you see `sim_best ≈ 0.10–0.20` on every turn despite a stable conversation, something volatile is sitting early in your prompt prefix — this proxy's approach of stripping it is the fix.
