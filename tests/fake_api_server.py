"""Local fake of the OpenAI and Anthropic HTTP APIs for tests (no network).

Replies are built with ``MockBackend`` (valid game JSON for any prompt) in the provider's
response shape. ``script`` holds (status, headers, body) answers used before the normal ones,
to test retries and errors. Every request is recorded (path, headers, JSON body).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from coop.llm import MockBackend


class FakeAPI:
    def __init__(self, delay_s: float = 0.0, cached: int = 0, reasoning: int = 7, stop: str = ""):
        self.requests, self.script = [], []
        self.delay_s, self.cached, self.reasoning, self.stop = delay_s, cached, reasoning, stop
        self.active = self.max_active = 0
        self.lock = threading.Lock()
        self.mock = MockBackend()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                    scripted = fake.script.pop(0) if fake.script else None
                    fake.active += 1
                    fake.max_active = max(fake.max_active, fake.active)
                try:
                    time.sleep(fake.delay_s)
                    if scripted is not None:
                        status, headers, payload = scripted
                    else:
                        status, headers, payload = 200, {}, fake.answer(self.path, body)
                    data = json.dumps(payload).encode()
                    self.send_response(status)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                finally:
                    with fake.lock:
                        fake.active -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def answer(self, path, body):
        if path.endswith("/messages"):
            msgs = [{"role": "system", "content": "".join(b["text"] for b in body.get("system", []))}] + body["messages"]
            text = self.mock.chat(msgs).text
            n_in = sum(len(m["content"]) for m in msgs) // 4
            return {"id": "msg_fake", "type": "message", "role": "assistant", "model": body["model"],
                    "content": [{"type": "thinking", "thinking": "", "signature": "x"},
                                {"type": "text", "text": text}],
                    "stop_reason": self.stop or "end_turn",
                    "usage": {"input_tokens": max(0, n_in - self.cached), "output_tokens": 30 + self.reasoning,
                              "cache_read_input_tokens": self.cached, "cache_creation_input_tokens": 0,
                              "output_tokens_details": {"thinking_tokens": self.reasoning}}}
        text = self.mock.chat(body["messages"]).text
        n_in = sum(len(m["content"]) for m in body["messages"]) // 4
        return {"id": "chatcmpl-fake", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "finish_reason": self.stop or "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": 30 + self.reasoning,
                          "prompt_tokens_details": {"cached_tokens": self.cached},
                          "completion_tokens_details": {"reasoning_tokens": self.reasoning}}}
