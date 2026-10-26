"""Windowing and batched inference shared by the transformer scripts (docs/TOXICITY.md).

A conversation is tokenized and cut into windows of up to 512 tokens (510 + special tokens) with a 64-token
overlap; at most MAX_WINDOWS windows are kept (about 2,000 tokens, which covers ~90% of conversations).
The conversation score is the maximum over its windows: a conversation is toxic if any part of it is.
"""

import numpy as np
import torch

MAX_WINDOWS = 4
WINDOW = 510
STRIDE = 446  # 64 tokens of overlap
MAX_CHARS = 12_000  # ~3,000 tokens; text beyond is never reached by MAX_WINDOWS windows


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def windows(tokenizer, text: str) -> list[list[int]]:
    ids = tokenizer(text[:MAX_CHARS], add_special_tokens=False, truncation=False)["input_ids"]
    out, start = [], 0
    while True:
        out.append(tokenizer.build_inputs_with_special_tokens(ids[start : start + WINDOW]))
        if start + WINDOW >= len(ids) or len(out) == MAX_WINDOWS:
            return out
        start += STRIDE


def pad(tokenizer, seqs: list[list[int]], multiple: int = 64):
    # round up to a multiple: few distinct shapes, so the MPS graph cache stays small
    n = min(-(-max(len(s) for s in seqs) // multiple) * multiple, WINDOW + 2)
    ids = torch.full((len(seqs), n), tokenizer.pad_token_id)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


@torch.no_grad()
def score_conversations(model, tokenizer, texts, logit_index=0, batch_windows=8, progress=None) -> np.ndarray:
    """Max-over-windows logit for each conversation."""
    dev = next(model.parameters()).device
    model.eval()
    owner, seqs = [], []
    for i, t in enumerate(texts):
        for w in windows(tokenizer, t):
            owner.append(i)
            seqs.append(w)
    order = np.argsort([len(s) for s in seqs])  # length-sorted batches waste less padding
    logits = np.empty(len(seqs))
    for b in range(0, len(order), batch_windows):
        idx = order[b : b + batch_windows]
        ids, mask = pad(tokenizer, [seqs[i] for i in idx])
        out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits[:, logit_index]
        logits[idx] = out.float().cpu().numpy()
        if dev.type == "mps":
            torch.mps.empty_cache()  # the MPS allocator otherwise keeps growing across batches (16 GB machine)
        if (b // batch_windows) % 50 == 0:
            if progress:
                progress(b, len(order))
    scores = np.full(len(texts), -np.inf)
    np.maximum.at(scores, np.array(owner), logits)
    return scores
