"""Fine-tune a conversation-level toxicity classifier on the ASE 2026 GitHub data (docs/TOXICITY.md).

Run from external/toxicity-gh with toxbert-venv (HF_HOME pointing at external/hf-cache):
    ../toxbert-venv/bin/python ../../backend/scripts/toxicity/train_transformer.py

Design
- Initialisation: unitary/unbiased-toxic-roberta (Detoxify, Jigsaw-trained). The new single-logit head is
  initialised from its "toxicity" output, so training starts from the zero-shot behaviour.
- Multiple-instance learning: a conversation is split into <= 4 windows of 512 tokens; its logit is the max over
  windows (toxic if any part is), trained with binary cross-entropy on the conversation label.
- Each epoch uses all toxic training conversations and NEG_RATIO times as many non-toxic ones, re-sampled per
  epoch. The decision threshold is chosen afterwards on the validation split at the natural rate.
- Model selection: validation average precision (natural rate). The test split is scored once, at the end.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import report  # noqa: E402
from windows import device, pad, score_conversations, windows  # noqa: E402

BASE_MODEL = "unitary/unbiased-toxic-roberta"
SEED, EPOCHS, NEG_RATIO, LR, CONV_BATCH, ACCUM = 13, 3, 10, 2e-5, 2, 8
W = Path("work")
OUT = W / "models" / "toxic-roberta-github"
OUT.mkdir(parents=True, exist_ok=True)
log = open(W / "train.log", "a")


def say(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dev = device()
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
zero = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
tox_idx = zero.config.label2id.get("toxicity", 0)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1, ignore_mismatched_sizes=True)
with torch.no_grad():  # start from the zero-shot "toxicity" output
    model.classifier.dense.load_state_dict(zero.classifier.dense.state_dict())
    model.classifier.out_proj.weight.copy_(zero.classifier.out_proj.weight[tox_idx : tox_idx + 1])
    model.classifier.out_proj.bias.copy_(zero.classifier.out_proj.bias[tox_idx : tox_idx + 1])
del zero
model.gradient_checkpointing_enable()
model.to(dev)

tr, va = pd.read_parquet(W / "large_train.parquet"), pd.read_parquet(W / "large_val.parquet")
pos, neg = tr[tr.label == 1], tr[tr.label == 0]
steps_per_epoch = (len(pos) * (1 + NEG_RATIO)) // (CONV_BATCH * ACCUM)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS)
loss_fn = torch.nn.BCEWithLogitsLoss()
config = {"base_model": BASE_MODEL, "seed": SEED, "epochs": EPOCHS, "neg_ratio": NEG_RATIO, "lr": LR,
          "conversation_batch": CONV_BATCH, "grad_accum": ACCUM, "train_toxic": len(pos), "device": str(dev)}  # fmt: skip
say(f"config {json.dumps(config)}")

best_ap, history = -1.0, []
for epoch in range(1, EPOCHS + 1):
    model.train()
    sample = pd.concat([pos, neg.sample(len(pos) * NEG_RATIO, random_state=SEED + epoch)]).sample(frac=1, random_state=epoch)
    texts, labels = sample.text.tolist(), sample.label.to_numpy(float)
    t0, running = time.time(), 0.0
    for step, start in enumerate(range(0, len(texts), CONV_BATCH), 1):
        conv_windows = [windows(tok, t) for t in texts[start : start + CONV_BATCH]]
        flat = [w for ws in conv_windows for w in ws]
        ids, mask = pad(tok, flat)
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits[:, 0]
        bounds = np.cumsum([0] + [len(ws) for ws in conv_windows])
        conv_logits = torch.stack([logits[bounds[i] : bounds[i + 1]].max() for i in range(len(conv_windows))])
        y = torch.tensor(labels[start : start + CONV_BATCH], dtype=torch.float32, device=dev)
        loss = loss_fn(conv_logits, y) / ACCUM
        loss.backward()
        running += loss.item() * ACCUM
        if step % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        if step % 200 == 0:
            done = start + CONV_BATCH
            say(f"epoch {epoch} {done}/{len(texts)} loss {running / 200:.4f} "
                f"{(time.time() - t0) / done * (len(texts) - done) / 60:.0f} min left in epoch")  # fmt: skip
            running = 0.0
    s_va = score_conversations(model, tok, va.text.tolist())
    ap = average_precision_score(va.label, s_va)
    history.append({"epoch": epoch, "val_average_precision": ap, "minutes": (time.time() - t0) / 60})
    say(f"epoch {epoch} done in {(time.time() - t0) / 60:.0f} min | validation AP {ap:.3f}")
    if ap > best_ap:
        best_ap = ap
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
        np.save(OUT / "val_scores.npy", s_va)
        say(f"saved best model (epoch {epoch})")

json.dump({"config": config, "history": history, "best_val_ap": best_ap}, open(OUT / "training.json", "w"), indent=2)

# Final, single evaluation on the untouched test split and the fully human-labelled Imran set.
best = AutoModelForSequenceClassification.from_pretrained(OUT).to(dev)
te, im = pd.read_parquet(W / "large_test.parquet"), pd.read_parquet(W / "imran_conversations.parquet")
s_va = np.load(OUT / "val_scores.npy")
s_te = score_conversations(best, tok, te.text.tolist())
s_im = score_conversations(best, tok, im.text.tolist())
pd.DataFrame({"issue_url": te.issue_url, "score": s_te}).to_parquet(W / "results/finetuned_test_scores.parquet")
res = report("Fine-tuned toxic RoBERTa (GitHub)", (va.label, s_va), (te.label, s_te, te.repo), (im.label, s_im),
             out_path=W / "results/finetuned.json")  # fmt: skip
json.dump({"threshold": res["threshold"]}, open(OUT / "threshold.json", "w"))
say(f"final: {json.dumps(res['test'])}")
