"""OpenAI emotions runner against a fake Responses API: protocol, parsing, refusals and the answer cache."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "app" / "tools" / "openai_emotions_runner.py"
EMOTIONS = ["anger", "joy", "love", "sadness", "fear", "surprise"]


class FakeAPI(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAPI.requests.append(body)
        text = body["input"]
        if "refuse" in text:
            content = [{"type": "refusal", "refusal": "I can't help with that."}]
        else:
            labels = {e: e in text for e in EMOTIONS}  # the "model" labels the emotion names it reads
            content = [{"type": "output_text", "text": json.dumps(labels)}]
        reply = {"status": "completed", "output": [{"type": "reasoning", "summary": []}, {"type": "message", "content": content}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def run(server, cache: Path, texts: list[str]) -> list[dict]:
    env = {"OPENAI_API_KEY": "test", "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "PATH": ""}
    proc = subprocess.Popen([sys.executable, str(RUNNER), "gpt-test", str(cache)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, env=env)  # fmt: skip
    assert proc.stdout.readline().strip() == "READY"
    out, _ = proc.communicate(json.dumps([{"id": i, "text": t} for i, t in enumerate(texts)]) + "\n", timeout=60)
    return json.loads(out)


def test_runner_parses_answers_skips_refusals_and_caches(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeAPI.requests = []
    try:
        texts = ["so much anger and joy here", "nothing to see", "please refuse this"]
        first = run(server, tmp_path / "cache.jsonl", texts)
        assert [r["emotions"] for r in first] == [["anger", "joy"], [], None]
        body = FakeAPI.requests[0]
        assert body["model"] == "gpt-test" and body["text"]["format"]["strict"] is True
        assert len(FakeAPI.requests) == 3

        # answers are cached; the refusal is not, so it is asked again
        second = run(server, tmp_path / "cache.jsonl", texts)
        assert second == first
        assert len(FakeAPI.requests) == 4 and FakeAPI.requests[-1]["input"] == "please refuse this"
    finally:
        server.shutdown()


SETTINGS_SCRIPT = """
from fastapi.testclient import TestClient
from app import api, emotion_settings
from app.stages.text import EmotionsStage

emotion_settings.check = lambda key, model: None  # no call to OpenAI
with TestClient(api.app) as c:
    assert c.get("/api/settings/emotions").json()["backend"] == "roberta"
    assert c.put("/api/settings/emotions", json={"backend": "openai"}).status_code == 422  # no key yet
    out = c.post("/api/settings/emotions/key", json={"key": "sk-test-abcd1234"}).json()
    assert out["key_last4"] == "1234" and out["key_source"] == "ui" and "sk-test" not in str(out)
    out = c.put("/api/settings/emotions", json={"backend": "openai", "model": "gpt-6-luna"}).json()
    assert out["backend"] == "openai" and out["model"] == "gpt-6-luna"
    assert EmotionsStage().tool_status() is None
    cfg = emotion_settings.current()
    assert cfg.api_key == "sk-test-abcd1234" and cfg.backend == "openai"
    out = c.delete("/api/settings/emotions/key").json()
    assert out["backend"] == "roberta" and out["key_last4"] is None
print("OK")
"""


def test_settings_endpoints_store_key_and_backend(tmp_path):
    # separate process: a fresh database in tmp_path, never the real one
    env = {"DATA_DIR": str(tmp_path), "PATH": ""}
    out = subprocess.run([sys.executable, "-c", SETTINGS_SCRIPT], cwd=Path(__file__).resolve().parents[1], env=env,
                         capture_output=True, text=True, timeout=60)  # fmt: skip
    assert out.stdout.strip().endswith("OK"), out.stderr[-2000:]
