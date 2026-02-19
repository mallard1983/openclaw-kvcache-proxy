"""
Replay captured requests through the optimization proxy, one at a time.
Reads request bodies from proxy_capture.log and POSTs them to the proxy.

Shows:
  - What normalization the proxy applied (ts/msg_id removals)
  - How long each request took
  - The model's response text

After each request, check llama-server console for:
  sim_best = X   (want this MUCH higher than the baseline 0.151)
  "forcing full prompt re-processing" (SWA limitation, may still appear)
  prompt eval time (want this lower on requests 2+)
"""

import json
import re
import time
import urllib.request
import urllib.error

LOG_FILE = r"C:\projects\openclawproxy\proxy_capture.log"
PROXY_URL = "http://localhost:1234/v1/responses"

# ── Extract request bodies from the log ──────────────────────────────────────

def extract_requests(log_path):
    """Parse all POST /v1/responses request bodies out of the capture log."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Each request block starts with the BODY: marker after a CATCH-ALL header
    # Split on the log timestamp + BODY: pattern
    body_pattern = re.compile(
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ BODY:\n(\{)',
        re.MULTILINE
    )

    requests = []
    matches = list(body_pattern.finditer(content))

    for i, m in enumerate(matches):
        # The JSON starts at the '{' — find where it ends by scanning forward
        start = m.start(1)
        # Find the next log timestamp line to know where the JSON ends
        if i + 1 < len(matches):
            # Search backwards from next match for end of JSON
            end_search = content[start:m.end(1) + 500000]  # up to 500KB
            next_ts = re.search(r'\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', end_search)
            if next_ts:
                json_str = end_search[:next_ts.start()]
            else:
                json_str = end_search
        else:
            json_str = content[start:]

        try:
            body = json.loads(json_str.strip())
            if "input" in body:  # only include actual /v1/responses requests
                requests.append(body)
        except json.JSONDecodeError:
            pass  # skip malformed blocks

    return requests


def summarize_input(input_items):
    """One-line summary of what's in the input array."""
    roles = []
    for item in input_items:
        t = item.get("type") or item.get("role", "?")
        roles.append(t)
    return f"{len(input_items)} items: [{', '.join(roles)}]"


def consume_sse_stream(response):
    """Read a streaming SSE response and collect the final text."""
    output_text = ""
    tool_calls = []
    usage = {}

    for line in response:
        line = line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str:
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        t = data.get("type", "")
        if t == "response.output_text.delta":
            output_text += data.get("delta", "")
        elif t == "response.completed":
            resp = data.get("response", {})
            usage = resp.get("usage", {})
            for item in resp.get("output", []):
                if item.get("type") == "function_call":
                    tool_calls.append(item.get("name", "?"))
                elif item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") == "output_text":
                            output_text = block.get("text", output_text)

    return output_text.strip(), tool_calls, usage


# ── Main replay loop ──────────────────────────────────────────────────────────

def main():
    print("Loading requests from log...")
    all_requests = extract_requests(LOG_FILE)
    print(f"Found {len(all_requests)} total captured requests\n")

    # Find the second conversation: look for the reset where input shrinks to 2 items
    # (system + new-session-greet only)
    conv2_start = None
    for i, req in enumerate(all_requests):
        inp = req.get("input", [])
        if len(inp) == 2:
            roles = [x.get("role") or x.get("type") for x in inp]
            if roles == ["system", "user"]:
                if i > 0:  # skip the very first request
                    conv2_start = i
                    break

    if conv2_start is None:
        print("Could not find a second conversation (session reset) in the log.")
        print("Replaying from request index 0 instead.")
        conv2_start = 0

    # Take 3 requests from conversation 2
    conv2 = all_requests[conv2_start:conv2_start + 3]
    print(f"Second conversation starts at captured request index {conv2_start}")
    print(f"Replaying {len(conv2)} requests:\n")

    for req_num, body in enumerate(conv2, 1):
        inp = body.get("input", [])
        print(f"{'='*60}")
        print(f"REQUEST {req_num}/{len(conv2)}")
        print(f"  Input:  {summarize_input(inp)}")
        print(f"  Stream: {body.get('stream', False)}")

        # Force streaming on so we get SSE back
        body["stream"] = True

        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            PROXY_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer llama",
                "Accept": "application/json",
            },
            method="POST",
        )

        print(f"\n  Sending to proxy... (waiting for response)")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                output_text, tool_calls, usage = consume_sse_stream(resp)
            elapsed = time.time() - t0

            print(f"  Done in {elapsed:.1f}s")
            print(f"  Usage:  {usage}")
            if tool_calls:
                print(f"  Tools called: {tool_calls}")
            print(f"\n  Model response:")
            # Print first 400 chars of response
            preview = output_text[:400]
            if len(output_text) > 400:
                preview += f"... [{len(output_text)} chars total]"
            for line in preview.splitlines():
                print(f"    {line}")

        except urllib.error.URLError as e:
            elapsed = time.time() - t0
            print(f"  ERROR after {elapsed:.1f}s: {e}")
            print("  Is the proxy running on port 1234?")

        print()
        if req_num < len(conv2):
            input(f"  >>> Press Enter to send request {req_num + 1}...")
            print()


if __name__ == "__main__":
    main()
