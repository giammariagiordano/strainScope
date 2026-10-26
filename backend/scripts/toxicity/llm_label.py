"""Label conversations with the ASE 2026 pipeline's LLM step: Ministral-3-14B through a local Ollama server, using the
package's "Toxicity and Category Explanation Generation Prompt" verbatim (backup/prompts/prompts.md). No text leaves
the machine.

Run from external/toxicity-gh with toxbert-venv, Ollama running and the model pulled (`ollama pull ministral-3:14b`):
    ../toxbert-venv/bin/python ../../backend/scripts/toxicity/llm_label.py work/strainscope_conversations.parquet \
        work/llm/strainscope.jsonl [--id-col id] [--limit N]

Output: one JSON line per conversation (id, toxicity_prediction, explanation_text, the eight event-category scores,
truncated, seconds), appended as it goes: an interrupted run resumes where it stopped.

Deviations from the package (which does not record its inference settings): temperature 0 and a fixed seed for
reproducibility; JSON-schema constrained output so every reply parses; conversations cut at MAX_CHARS so prompt +
conversation fit in NUM_CTX tokens (flagged as truncated).
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

OLLAMA = "http://localhost:11434/api/generate"
PROMPTS = Path("backup/prompts/prompts.md")
NUM_CTX = 8192
MAX_CHARS = 20_000  # ~5,000 tokens + ~1,900 for the prompt; covers > 99% of StrainScope conversations
SEED = 13
CATEGORIES = ["person_hostility", "process_hostility", "dismissiveness", "escalation_without_explicit_abuse",
              "subtle_criticism", "hostility_density", "resolution_language", "behavior_callout"]  # fmt: skip
SCHEMA = {
    "type": "object",
    "properties": {
        "toxicity_prediction": {"type": "integer", "enum": [0, 1]},
        "explanation_text": {"type": "string"},
        "event_category_scores": {
            "type": "object",
            "properties": {c: {"type": "integer", "minimum": 0, "maximum": 10} for c in CATEGORIES},
            "required": CATEGORIES,
        },
    },
    "required": ["toxicity_prediction", "explanation_text", "event_category_scores"],
}


def category_prompt() -> str:
    text = PROMPTS.read_text()
    m = re.search(r"## Toxicity and Category Explanation Generation Prompt\n(.*?)\n## ", text, re.S)
    return m.group(1).strip()


def ask(model: str, prompt: str) -> dict:
    body = {"model": model, "prompt": prompt, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0, "seed": SEED, "num_ctx": NUM_CTX}}  # fmt: skip
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        reply = json.loads(r.read())
    return json.loads(reply["response"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model", default="ministral-3:14b")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    template = category_prompt()
    df = pd.read_parquet(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = df[~df[args.id_col].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already labelled, {len(todo)} to go ({args.model})", flush=True)

    t0 = time.time()
    with out.open("a") as f:
        for n, (cid, text) in enumerate(zip(todo[args.id_col], todo[args.text_col]), 1):
            t = time.time()
            prompt = template.format(conversation=text[:MAX_CHARS])
            for attempt in range(2):
                try:
                    res = ask(args.model, prompt)
                    break
                except (json.JSONDecodeError, KeyError) as e:
                    if attempt:
                        print(f"   {cid}: unparseable reply ({e}), skipped", flush=True)
                        res = None
            if res is None:
                continue
            row = {"id": int(cid), "toxicity_prediction": res["toxicity_prediction"],
                   "explanation_text": res["explanation_text"], **res["event_category_scores"],
                   "truncated": len(text) > MAX_CHARS, "seconds": round(time.time() - t, 1)}  # fmt: skip
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"   {n}/{len(todo)}  {rate:.1f}s/conv  ETA {rate * (len(todo) - n) / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
