#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Generator, Optional, Tuple

import requests


"""
## Server API Reference

- **Base URL**: `http://localhost:8100`
- **Content-Type**: `application/json` (requests), `application/json` (responses), `application/x-ndjson` (streaming)
- **Auth**: None
- **Common header**: `X-Session-Id` (optional on request; echoed/created on response for multi-turn endpoints)

### Models
```json
{
  "prompt": "string"
}
```

## VTC APIs

### POST `/vtc/chat/v{ver_no}/single`
- **Body**: `{"prompt": "..."}`  
- **Response**: `{"response": "..."}`

```bash
curl -X POST http://localhost:8100/vtc/chat/v1/single \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello"}'
```

### POST `/vtc/stream/v{ver_no}/single`
- **Body**: `{"prompt": "..."}`  
- **Response**: NDJSON stream; each line is `{"response": "<chunk>"}`  
- **Media type**: `application/x-ndjson`

```bash
curl -N http://localhost:8100/vtc/stream/v1/single \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Stream this"}'
```

### POST `/vtc/chat/v{ver_no}/multi`
- **Headers (request)**: optional `X-Session-Id`  
- **Headers (response)**: `X-Session-Id` (generated if not provided)  
- **Body**: `{"prompt": "..."}`  
- **Response**: `{"response": "..."}`

```bash
# First turn (no session id provided)
curl -i -X POST http://localhost:8100/vtc/chat/v1/multi \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Who are you?"}'

# Subsequent turn (reusing session id)
curl -X POST http://localhost:8100/vtc/chat/v1/multi \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: <SESSION_ID_FROM_RESPONSE_HEADER>" \
  -d '{"prompt":"What can you do?"}'
```

## Zephyr APIs

### POST `/zephyr/chat/v{ver_no}/multi`
- **Version**: `ver_no` must be `1` or `2`  
- **Headers (request)**: optional `X-Session-Id`  
- **Headers (response)**: `X-Session-Id` (generated if not provided)  
- **Body**: `{"prompt": "..."}`  
- **Response**: `{"response": "..."}` or `{"error": "Invalid version number"}`

```bash
curl -i -X POST http://localhost:8100/zephyr/chat/v2/multi \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Summarize Zephyr"}'
```

### GET `/zephyr/reset`
- **Purpose**: Clear chat history for the provided session  
- **Headers (request)**: optional `X-Session-Id`  
- **Response**: `{"response": "OK"}`

```bash
curl -X GET http://localhost:8100/zephyr/reset \
  -H "X-Session-Id: <SESSION_ID>"
```

### Notes
- For VTC endpoints, `ver_no` is an integer.
- For Zephyr, only versions `1` and `2` are supported; other values return an error payload.

- Added a concise Markdown reference for all routes in `src/server.py`.
- Covered request/response schemas, headers, streaming format, versioning, and curl examples.
"""



class ApiError(Exception):
    pass


class VtcChatClient:
    def __init__(self, base_url: str = "http://localhost:8100", ver_no: int = 1, session_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.ver_no = ver_no
        self.session_id = session_id

    def chat_single(self, prompt: str, timeout: float = 60.0) -> str:
        url = f"{self.base_url}/vtc/chat/v{self.ver_no}/single"
        resp = requests.post(url, json={"prompt": prompt}, timeout=timeout)
        self._raise_for_response(resp)
        data = resp.json()
        return data.get("response", "")

    def chat_stream(self, prompt: str, timeout: float = 300.0) -> Generator[str, None, None]:
        """
        Streams NDJSON from /vtc/stream/v{ver_no}/single.
        Yields text chunks as they arrive.
        """
        url = f"{self.base_url}/vtc/stream/v{self.ver_no}/single"
        with requests.post(url, json={"prompt": prompt}, stream=True, timeout=timeout) as resp:
            self._raise_for_response(resp)
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunk = obj.get("response")
                    if chunk:
                        yield chunk
                except json.JSONDecodeError:
                    # Safeguard: if a line isn't valid JSON, skip it
                    continue

    def chat_multi(self, prompt: str, timeout: float = 60.0) -> Tuple[str, str]:
        """
        Sends/maintains X-Session-Id automatically. Returns (response_text, session_id).
        """
        url = f"{self.base_url}/vtc/chat/v{self.ver_no}/multi"
        headers = {}
        if self.session_id:
            headers["X-Session-Id"] = self.session_id

        resp = requests.post(url, json={"prompt": prompt}, headers=headers, timeout=timeout)
        self._raise_for_response(resp)

        # Capture/refresh session_id from response header
        new_session_id = resp.headers.get("X-Session-Id")
        if new_session_id:
            self.session_id = new_session_id

        data = resp.json()
        return data.get("response", ""), self.session_id or ""

    @staticmethod
    def _raise_for_response(resp: requests.Response) -> None:
        if resp.status_code != 200:
            try:
                payload = resp.json()
            except Exception:
                payload = resp.text
            raise ApiError(f"HTTP {resp.status_code}: {payload}")


class ZephyrChatClient:
    def __init__(self, base_url: str = "http://localhost:8100", ver_no: int = 1, session_id: Optional[str] = None):
        if ver_no not in (1, 2):
            raise ValueError("Zephyr ver_no must be 1 or 2")
        self.base_url = base_url.rstrip("/")
        self.ver_no = ver_no
        self.session_id = session_id

    def chat(self, prompt: str, timeout: float = 60.0) -> Tuple[str, str]:
        """
        Sends/maintains X-Session-Id automatically. Returns (response_text, session_id).

        Returns:
            tuple[str, str]: (response_text, session_id)
        """
        
        url = f"{self.base_url}/zephyr/chat/v{self.ver_no}/multi"
        headers = {
            "Content-Type": "application/json",
        }
        if self.session_id:
            headers["X-Session-Id"] = self.session_id

        resp = requests.post(url, json={"prompt": prompt}, headers=headers, timeout=timeout)
        self._raise_for_response(resp)

        new_session_id = resp.headers.get("X-Session-Id")
        if new_session_id:
            self.session_id = new_session_id

        data = resp.json()
        if "error" in data:
            raise ApiError(f"Zephyr error: {data['error']}")
        return data.get("response", ""), self.session_id or ""

    def reset(self, timeout: float = 15.0) -> bool:
        """
        Clears chat history for the current session (if any).
        """
        url = f"{self.base_url}/zephyr/reset"
        headers = {}
        if self.session_id:
            headers["X-Session-Id"] = self.session_id

        resp = requests.get(url, headers=headers, timeout=timeout)
        self._raise_for_response(resp)
        data = resp.json()
        return data.get("response") == "OK"

    @staticmethod
    def _raise_for_response(resp: requests.Response) -> None:
        if resp.status_code != 200:
            try:
                payload = resp.json()
            except Exception:
                payload = resp.text
            raise ApiError(f"HTTP {resp.status_code}: {payload}")


# Example usage
if __name__ == "__main__":
    # VTC single turn
    vtc = VtcChatClient(base_url="http://localhost:8100", ver_no=1)
    print("[VTC single] ->", vtc.chat_single("Hello!"))

    # VTC streaming
    print("[VTC stream] ->", end=" ", flush=True)
    for chunk in vtc.chat_stream("Stream me a response, please."):
        print(chunk, end="", flush=True)
    print()

    # VTC multi-turn (session automatically managed)
    vtc_multi = VtcChatClient(base_url="http://localhost:8100", ver_no=1)
    r1, sid = vtc_multi.chat_multi("Who are you?")
    print(f"[VTC multi 1] (sid={sid}) ->", r1)
    r2, sid = vtc_multi.chat_multi("And what can you do?")
    print(f"[VTC multi 2] (sid={sid}) ->", r2)

    # Zephyr multi-turn (v1 or v2)
    zephyr = ZephyrChatClient(base_url="http://localhost:8100", ver_no=2)
    z1, zsid = zephyr.chat("Summarize the purpose of Zephyr.")
    print(f"[Zephyr 1] (sid={zsid}) ->", z1)
    z2, zsid = zephyr.chat("Now refine that summary.")
    print(f"[Zephyr 2] (sid={zsid}) ->", z2)

    # Optional: reset Zephyr session
    if zephyr.reset():
        print("[Zephyr] session reset OK")