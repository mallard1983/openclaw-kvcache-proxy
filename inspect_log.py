import re
import json

LOG_FILE = r"C:\projects\openclawproxy\proxy_capture.log"

with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
    content = f.read()

# Split on the separator lines for each request
requests = re.split(r'CATCH-ALL: POST /v1/responses', content)

print(f"Total request blocks found: {len(requests) - 1}\n")

for i, block in enumerate(requests[1:], 1):
    # Get timestamp from HEADERS line
    ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', block)
    ts = ts_match.group(1) if ts_match else "?"
    
    # Get content-length
    cl_match = re.search(r"'content-length': '(\d+)'", block)
    cl = cl_match.group(1) if cl_match else "?"
    
    # Try to find the start of the JSON body and count input items
    body_match = re.search(r'BODY:\n(\{.*?)(?=\n\d{4}-\d{2}-\d{2}|$)', block, re.DOTALL)
    n_items = "?"
    has_tool_calls = False
    if body_match:
        try:
            body = json.loads(body_match.group(1))
            items = body.get("input", [])
            n_items = len(items)
            has_tool_calls = any(
                item.get("type") in ("function_call", "function_call_output")
                for item in items
            )
        except Exception:
            pass
    
    print(f"Request {i:02d} | {ts} | content-length={cl:>7} | input_items={n_items} | tool_calls={has_tool_calls}")
