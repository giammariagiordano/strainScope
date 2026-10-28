"""Emotions runner (OpenAI API, e.g. gpt-6-luna). Executed with the backend's Python (only httpx is needed).

Protocol: the same as emotions_runner.py. One JSON list per line on stdin, [{"id": ..., "text": ...}]; one JSON list
per line on stdout, [{"id": ..., "emotions": ["anger", ...]}] (empty list: no emotion; null: the API refused or
failed, so the text stays unclassified and is retried on the next run). "READY" is printed at start-up.

    python openai_emotions_runner.py <model> <cache.jsonl> [examples.json]

Unlike the local classifiers, texts are SENT TO OPENAI. Each text is classified on its own (no context from other
texts), with a fixed prompt built on Shaver's taxonomy and, optionally, labelled examples taken from training data
only. The model's answer is constrained to a JSON schema (structured outputs). Every answer is appended to the cache,
keyed by model, prompt version and text, so a text is never paid twice and every label can be traced back to the
model's raw answer (replicability). OPENAI_API_KEY must be set; OPENAI_CONCURRENCY (default 8) parallel requests.
"""

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

EMOTIONS = ["anger", "joy", "love", "sadness", "fear", "surprise"]
PROMPT_VERSION = "1"  # change whenever the prompt or the examples change: the cache key depends on it
URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/responses"
PROMPT = """You annotate emotions in messages written by developers on GitHub (issues, pull requests, code reviews).
Use Shaver et al.'s six basic emotions. A message can express none, one or several of them.

- anger: anger, irritation, annoyance, frustration, exasperation, disgust, rage.
- joy: happiness, cheerfulness, satisfaction, enthusiasm, relief, pride, optimism.
- love: affection, appreciation, gratitude towards someone, caring, admiration, liking.
- sadness: disappointment, regret, shame, sorrow, discouragement, feeling sorry.
- fear: worry, anxiety, nervousness, being afraid or unsure of consequences.
- surprise: amazement, astonishment, unexpected realization, confusion at something unexpected.

Label only the emotions the author expresses in the message, not emotions the author describes in others or that
the topic might evoke. Technical statements without emotional content get no emotion. Code, quotes and mentions
have been removed from the message."""

model = sys.argv[1]
cache_path = Path(sys.argv[2])
examples = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else []
key = os.environ.get("OPENAI_API_KEY")
if not key:
    sys.exit("OPENAI_API_KEY is not set")

instructions = PROMPT
if examples:
    shots = "\n\n".join(f"Message: {e['text']}\nEmotions: {', '.join(e['emotions']) or 'none'}" for e in examples)
    instructions += "\n\nAnnotated examples:\n\n" + shots
SCHEMA = {
    "type": "object",
    "properties": {e: {"type": "boolean"} for e in EMOTIONS},
    "required": EMOTIONS,
    "additionalProperties": False,
}


def cache_key(text: str) -> str:
    return hashlib.sha256(f"{model}\0{PROMPT_VERSION}\0{instructions}\0{text}".encode()).hexdigest()


cache: dict[str, list[str] | None] = {}
if cache_path.exists():
    for line in cache_path.read_text().splitlines():
        entry = json.loads(line)
        cache[entry["key"]] = entry["emotions"]
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache_file = cache_path.open("a")
lock = threading.Lock()
client = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {key}"})


def ask(text: str) -> list[str] | None:
    body = {
        "model": model,
        "instructions": instructions,
        "input": text,
        "text": {"format": {"type": "json_schema", "name": "emotions", "strict": True, "schema": SCHEMA}},
    }
    for attempt in range(6):
        try:
            r = client.post(URL, json=body)
        except httpx.TransportError:
            r = None
        if r is not None and r.status_code == 200:
            break
        if r is not None and r.status_code not in (408, 409, 429) and r.status_code < 500:
            raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")
        time.sleep(min(60, 2**attempt))
    else:
        return None
    reply = r.json()
    for item in reply.get("output", []):
        if item.get("type") != "message":
            continue  # e.g. reasoning items
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                labels = json.loads(part["text"])
                return [e for e in EMOTIONS if labels[e]]
    return None  # refusal or incomplete answer: no label rather than a guessed one


def classify(text: str) -> list[str] | None:
    k = cache_key(text)
    if k in cache:
        return cache[k]
    emotions = ask(text)
    if emotions is not None:  # failures are not cached, so they are retried
        with lock:
            cache[k] = emotions
            cache_file.write(json.dumps({"key": k, "model": model, "prompt": PROMPT_VERSION, "emotions": emotions}) + "\n")
            cache_file.flush()
    return emotions


pool = ThreadPoolExecutor(int(os.environ.get("OPENAI_CONCURRENCY", "8")))
print("READY", flush=True)
for line in sys.stdin:
    batch = json.loads(line)
    results = list(pool.map(classify, [b["text"] for b in batch]))
    print(json.dumps([{"id": b["id"], "emotions": e} for b, e in zip(batch, results)]), flush=True)
