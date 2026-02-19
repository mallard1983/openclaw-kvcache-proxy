"""
LLM Logging Proxy
Sits between OpenClaw (port 1234) and llama-server (port 12345)
Logs all traffic for analysis without modifying anything.
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import logging
import time
from datetime import datetime

# --- Config ---
LISTEN_PORT = 1234
BACKEND_URL = "http://localhost:12345"
LOG_FILE = "proxy_capture.log"

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Logging Proxy")

def log_separator(label: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"  {label}  [{datetime.now().isoformat()}]")
    logger.info(f"{'='*60}")

def summarize_messages(messages: list) -> dict:
    """Pull out key structural info for quick analysis."""
    summary = {
        "total_messages": len(messages),
        "roles": [m.get("role") for m in messages],
        "system_prompt_length": None,
        "has_tool_calls": False,
        "has_tool_results": False,
        "tool_definitions_count": 0,
    }
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            summary["system_prompt_length"] = len(content) if isinstance(content, str) else "complex"
        if m.get("role") == "assistant" and m.get("tool_calls"):
            summary["has_tool_calls"] = True
        if m.get("role") == "tool":
            summary["has_tool_results"] = True
    return summary

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    log_separator("INCOMING REQUEST")
    
    # Structural summary first for quick reads
    messages = body.get("messages", [])
    summary = summarize_messages(messages)
    logger.info(f"STRUCTURE SUMMARY:\n{json.dumps(summary, indent=2)}")
    
    # Tool definitions if present
    if body.get("tools"):
        summary["tool_definitions_count"] = len(body["tools"])
        tool_names = [t.get("function", {}).get("name", "unknown") for t in body["tools"]]
        logger.info(f"TOOLS DEFINED: {tool_names}")

    # Full request body
    logger.info(f"FULL REQUEST BODY:\n{json.dumps(body, indent=2)}")

    start_time = time.time()

    if stream:
        # Handle streaming
        async def stream_generator():
            chunks = []
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST",
                    f"{BACKEND_URL}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
                            chunks.append(line)

            elapsed = time.time() - start_time
            log_separator(f"STREAMING RESPONSE (completed in {elapsed:.2f}s)")
            logger.info(f"TOTAL CHUNKS: {len(chunks)}")
            # Log full stream for analysis
            for chunk in chunks:
                logger.info(chunk)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        # Handle non-streaming
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{BACKEND_URL}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"}
            )

        elapsed = time.time() - start_time
        resp_json = resp.json()

        log_separator(f"RESPONSE (completed in {elapsed:.2f}s)")
        
        # Log usage stats if present
        if "usage" in resp_json:
            logger.info(f"TOKEN USAGE: {json.dumps(resp_json['usage'], indent=2)}")
        
        logger.info(f"FULL RESPONSE:\n{json.dumps(resp_json, indent=2)}")

        return JSONResponse(content=resp_json)


@app.get("/v1/models")
async def proxy_models(request: Request):
    """Forward model list requests transparently."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BACKEND_URL}/v1/models")
    return JSONResponse(content=resp.json())


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND_URL, "listen_port": LISTEN_PORT}


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(full_path: str, request: Request):
    """Catch any path we haven't explicitly handled - log and forward."""
    body = None
    try:
        body = await request.json()
    except Exception:
        body = await request.body()
        body = body.decode("utf-8", errors="replace") if body else None

    log_separator(f"CATCH-ALL: {request.method} /{full_path}")
    logger.info(f"METHOD: {request.method}")
    logger.info(f"PATH: /{full_path}")
    logger.info(f"HEADERS: {dict(request.headers)}")
    if body:
        logger.info(f"BODY:\n{json.dumps(body, indent=2) if isinstance(body, dict) else body}")

    # Forward to backend
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.request(
            method=request.method,
            url=f"{BACKEND_URL}/{full_path}",
            json=body if isinstance(body, dict) else None,
            content=body.encode() if isinstance(body, str) else None,
            headers={"Content-Type": request.headers.get("content-type", "application/json")}
        )

    log_separator(f"CATCH-ALL RESPONSE: {resp.status_code}")
    try:
        resp_json = resp.json()
        logger.info(json.dumps(resp_json, indent=2))
        return JSONResponse(content=resp_json, status_code=resp.status_code)
    except Exception:
        logger.info(f"RAW RESPONSE: {resp.text}")
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=resp.text, status_code=resp.status_code)


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting proxy on port {LISTEN_PORT} -> {BACKEND_URL}")
    logger.info(f"Capturing logs to: {LOG_FILE}")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)