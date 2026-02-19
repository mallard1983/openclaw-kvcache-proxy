"""
OpenClaw → llama-server Optimization Proxy
Listen: port 1234  →  Forward: http://localhost:12345

Normalizes requests to maximize KV cache prefix stability, addressing
the root cause of sim_best ≈ 0.15 observed in llama-server logs.

── Root cause analysis ──────────────────────────────────────────────────────
OpenClaw injects volatile fields into every request:

  1. SYSTEM PROMPT — "Inbound Context" section (~2500 tokens in):
         "message_id": "775b2410-..."   ← UUID changes every turn
     This sits early in the 50KB system prompt. LCP collapses to ~15%
     because everything after this point is treated as "new" by the cache.

  2. USER MESSAGES — timestamp prefix:
         [Wed 2026-02-18 20:48 UTC] Hello
     Injected per-message, busts the conversation-history prefix too.

── Optimizations applied ────────────────────────────────────────────────────
  STRIP_MESSAGE_IDS = True   (primary, high impact)
    Removes "message_id": "..." from all JSON blocks in system + user items.
    Expected: sim_best rises from ~0.15 toward ~0.95+

  STRIP_TIMESTAMPS = True    (secondary, lower impact)
    Removes [Day YYYY-MM-DD HH:MM UTC] prefix from user message text.

Both can be toggled below. Logging shows per-request normalization stats.
"""

import re
import json
import time
import copy
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

# ── Config ────────────────────────────────────────────────────────────────────
LISTEN_PORT = 1234
BACKEND_URL = "http://localhost:12345"
LOG_FILE = "proxy.log"

# Primary fix: strip volatile message_id UUIDs from JSON metadata blocks.
# Present in system prompt (inbound context) and each user message wrapper.
STRIP_MESSAGE_IDS = True

# Secondary fix: strip [Day YYYY-MM-DD HH:MM UTC] from user message text.
STRIP_TIMESTAMPS = True

# ── Patterns ─────────────────────────────────────────────────────────────────
# Matches the "message_id" line inside any JSON block (with optional trailing comma)
_MSG_ID_RE = re.compile(
    r'\n[ \t]*"message_id"\s*:\s*"[^"]+",?'
)

# Matches OpenClaw's per-message timestamp: [Wed 2026-02-18 20:48 UTC]
_TIMESTAMP_RE = re.compile(
    r'\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\] '
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="OpenClaw→llama Optimization Proxy")


def _strip_text(text: str) -> tuple[str, int, int]:
    """Apply all enabled normalizations to a text string.

    Returns (normalized_text, ts_removed, msg_ids_removed).
    """
    ts_n = msg_id_n = 0

    if STRIP_TIMESTAMPS:
        text, ts_n = _TIMESTAMP_RE.subn("", text)

    if STRIP_MESSAGE_IDS:
        text, msg_id_n = _MSG_ID_RE.subn("", text)

    return text, ts_n, msg_id_n


def normalize_input(input_items: list) -> tuple[list, dict]:
    """Return a normalized copy of the input array and a stats dict.

    Handles three item shapes:
      - role="system",    content: str          → normalize system prompt text
      - role="user",      content: [{type, text}] → normalize user message text
      - everything else                           → pass through unchanged
    """
    items = copy.deepcopy(input_items)
    stats = {"ts_removed": 0, "msg_ids_removed": 0, "items_modified": 0}

    for item in items:
        modified = False

        role = item.get("role")
        content = item.get("content")

        # System prompt: content is a plain string
        if role == "system" and isinstance(content, str):
            new_text, ts_n, mid_n = _strip_text(content)
            if new_text != content:
                item["content"] = new_text
                stats["ts_removed"] += ts_n
                stats["msg_ids_removed"] += mid_n
                modified = True

        # User messages: content is a list of content blocks
        elif role == "user" and isinstance(content, list):
            for block in content:
                if block.get("type") == "input_text" and "text" in block:
                    original = block["text"]
                    new_text, ts_n, mid_n = _strip_text(original)
                    if new_text != original:
                        block["text"] = new_text
                        stats["ts_removed"] += ts_n
                        stats["msg_ids_removed"] += mid_n
                        modified = True

        if modified:
            stats["items_modified"] += 1

    return items, stats


# ── Route handlers ────────────────────────────────────────────────────────────

@app.post("/v1/responses")
async def proxy_responses(request: Request):
    """Normalize then forward a Responses API request."""
    body = await request.json()

    original_input = body.get("input", [])
    normalized_input, stats = normalize_input(original_input)

    # Count input tokens from last completed response for logging context
    n_items = len(original_input)
    is_stream = body.get("stream", False)

    log.info(
        "POST /v1/responses | items=%d | ts_removed=%d | msg_ids_removed=%d | "
        "items_modified=%d | stream=%s",
        n_items,
        stats["ts_removed"],
        stats["msg_ids_removed"],
        stats["items_modified"],
        is_stream,
    )

    # Warn if no normalizations applied (cache will still miss)
    if stats["ts_removed"] == 0 and stats["msg_ids_removed"] == 0:
        log.warning("  → no volatile fields found; prompt sent as-is")

    modified_body = {**body, "input": normalized_input}

    if is_stream:
        return StreamingResponse(
            _stream_forward("/v1/responses", modified_body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{BACKEND_URL}/v1/responses",
            json=modified_body,
            headers={"Content-Type": "application/json"},
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _stream_forward(path: str, body: dict):
    """Pass the SSE byte stream from llama-server through verbatim.

    Using aiter_bytes() instead of aiter_lines() preserves the exact SSE
    framing (including blank-line event separators). Parsing lines and
    re-joining with \\n\\n splits 'event:' and 'data:' fields into separate
    events, causing clients to receive empty data and fail JSON.parse("").
    """
    t0 = time.time()
    bytes_sent = 0
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{BACKEND_URL}{path}",
            json=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
                bytes_sent += len(chunk)
    log.info("  → stream done in %.1fs, %d bytes", time.time() - t0, bytes_sent)


@app.get("/v1/models")
async def proxy_models():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BACKEND_URL}/v1/models")
    return JSONResponse(content=resp.json())


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": BACKEND_URL,
        "strip_timestamps": STRIP_TIMESTAMPS,
        "strip_message_ids": STRIP_MESSAGE_IDS,
    }


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def catch_all(full_path: str, request: Request):
    """Transparent passthrough for any path not explicitly handled."""
    body = None
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        body = raw.decode("utf-8", errors="replace") if raw else None

    log.info("passthrough: %s /%s", request.method, full_path)

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.request(
            method=request.method,
            url=f"{BACKEND_URL}/{full_path}",
            json=body if isinstance(body, dict) else None,
            content=body.encode() if isinstance(body, str) else None,
            headers={
                "Content-Type": request.headers.get("content-type", "application/json")
            },
        )

    try:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=resp.text, status_code=resp.status_code)


if __name__ == "__main__":
    import uvicorn

    log.info("Starting optimization proxy: port %d → %s", LISTEN_PORT, BACKEND_URL)
    log.info(
        "  strip_timestamps=%s  strip_message_ids=%s",
        STRIP_TIMESTAMPS,
        STRIP_MESSAGE_IDS,
    )
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
