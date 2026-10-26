"""EmoTxt (Calefato, Lanubile, Novielli, ACII 2017) features, reimplemented from its Java source
(external/emotxt/java/src: computing/Grams.java, computing/TermFrequency_InverseDocumentFrequency.java,
replacing/ReplacerTextWithMarks.java), quirks included, and checked identical to the Java output
(backend/scripts/emotions/train_emotxt.py). The Java code computes the IDF of each dictionary term by re-reading
every document (hours on 4,800 texts); here it takes seconds.

Input: texts already pre-processed by EmoTxt's Java tokenizer (lower-cased Stanford PTB tokens, URLs, mentions
and special strings removed), one per document.

Features, in this order:
- unigrams and bigrams of the dictionary: tf · idf, tf = count / tokens in the document, idf = log10(N / df);
- WordNet Affect categories: each token is replaced by its category (neutral, ambiguous, positive, negative lists,
  first match wins), then tf · idf of the categories in the replaced text.
"""

import csv
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

WORDNET_FILES = ("neutral_emotion.csv", "ambiguos-emotion.csv", "positive_emotion.csv", "negative_emotion.csv")
WORDNET_COLUMNS_ORDER = ("positive_emotion.csv", "negative_emotion.csv", "neutral_emotion.csv", "ambiguos-emotion.csv")
_WS = re.compile(r"\s+")


def java_split(text: str) -> list[str]:
    """Java's String.split("\\s+"): a leading empty string is kept, trailing ones dropped; "" gives [""]."""
    parts = _WS.split(text)
    while len(parts) > 1 and parts[-1] == "":
        parts.pop()
    return parts


def ngrams(tokens: list[str], n: int) -> list[str]:
    """CoreNLP StringUtils.getNgrams(tokens, n, n); bigrams joined with "_" as in Grams.getNgrams."""
    if n == 1:
        return list(tokens)
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def tf(grams: list[str]) -> dict[str, float]:
    total = len(grams)
    return {g: c / total for g, c in Counter(grams).items()} if total else {}


def build_dictionary(texts: list[str], n: int) -> list[str]:
    """Grams.getPositionWordMap on pre-processed texts: first ";"-separated field, lower-cased, tokens starting with
    "@" dropped (removeUserMention), n-grams other than "?" and "!", sorted. URLs were already removed by the
    pre-processing; the Java code's second pass is checked to change nothing (train_emotxt.py)."""
    ids: dict[str, str] = {}
    values: Counter = Counter()  # the map's current values (ids), for the Java check below
    i = 0
    for text in texts:
        line = text.split(";")[0].lower()
        tokens = [t for t in java_split(line) if t and t[0] != "@"]
        if not tokens:
            continue
        for g in ngrams(tokens, n):
            # Java tests map.containsValue(g), i.e. g against the ids, not the terms: a term already present is
            # re-numbered, and a number equal to an id already assigned (e.g. "18") never enters the dictionary.
            if g in ("?", "!") or values[g]:
                continue
            if g in ids:
                values[ids[g]] -= 1
            ids[g] = str(i)
            values[ids[g]] += 1
            i += 1
    return sorted(ids, key=lambda t: t.encode("utf-16-be"))  # Java String order (UTF-16 code units)


def write_dictionary(terms: list[str], ngram_dir: Path, kind: str) -> None:
    """{Unigrams,Bigrams}List_{1,2}.txt as PrintingFile.printNgrams: ids in sorted order, split after half."""
    prefix = "uni" if kind == "Unigrams" else "bi"
    half = len(terms) // 2
    ngram_dir.mkdir(parents=True, exist_ok=True)
    with open(ngram_dir / f"{kind}List_1.txt", "w", encoding="utf-8") as f1, open(ngram_dir / f"{kind}List_2.txt", "w", encoding="utf-8") as f2:
        for j, t in enumerate(terms):
            (f2 if j > half else f1).write(f"{t}\t{prefix}{j}\n")


def read_dictionary(ngram_dir: Path, kind: str) -> list[tuple[str, str]]:
    """[(term, column)] from {Unigrams,Bigrams}List_{1,2}.txt (the sorted dictionary split in two halves)."""
    out = []
    for half in (1, 2):
        for line in (ngram_dir / f"{kind}List_{half}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
    return out


def read_wordnet(res_dir: Path) -> dict[str, dict[str, list[str]]]:
    """{file: {category: [terms]}} in file and column order, as ReadingCSV.read_AllColumn_CSV (EXCEL format)."""
    lists = {}
    for name in WORDNET_FILES:
        with open(res_dir / name, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter=";"))
        header, body = rows[0], rows[1:]
        lists[name] = {h: [r[i] if i < len(r) else "" for r in body] for i, h in enumerate(header)}
    return lists


def marks_index(lists: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, set[str]]]:
    return {f: {m: set(terms) for m, terms in cats.items()} for f, cats in lists.items()}


def replace_with_marks(text: str, lists: dict[str, dict[str, list[str]]], sets: dict | None = None) -> str:
    """ReplacerTextWithMarks.replaceTermsWithMarks: " " + mark for every token found in a list."""
    sets = sets if sets is not None else marks_index(lists)
    out = ""
    for term in java_split(text):
        if term == "":
            continue
        for name in WORDNET_FILES:
            mark = None
            for m, terms in sets[name].items():
                if m.lower() == term:
                    mark = m.lower()
                    break
                if term in terms:
                    mark = m
                    break
            if mark is not None:
                out += " " + mark
                break
    return out


class EmoTxtFeatures:
    def __init__(self, ngram_dir: Path | None, wordnet_dir: Path, idf: dict[str, dict[str, float]] | None = None,
                 texts: list[str] | None = None):
        """Dictionaries from EmoTxt's n-gram files (`ngram_dir`) or built from training `texts`."""
        if texts is not None:
            self.unigrams = [(t, f"uni{j}") for j, t in enumerate(build_dictionary(texts, 1))]
            self.bigrams = [(t, f"bi{j}") for j, t in enumerate(build_dictionary(texts, 2))]
        else:
            self.unigrams = read_dictionary(ngram_dir, "Unigrams")
            self.bigrams = read_dictionary(ngram_dir, "Bigrams")
        lists = read_wordnet(wordnet_dir)
        self.lists = lists
        self.sets = marks_index(lists)
        self.categories = [m for name in WORDNET_COLUMNS_ORDER for m in lists[name]]
        self.idf = idf  # {"unigrams": {term: idf}, "bigrams": {...}, "wordnet": {...}}

    @property
    def columns(self) -> list[str]:
        return [c for _, c in self.unigrams] + [c for _, c in self.bigrams] + list(self.categories)

    def fit_idf(self, texts: list[str]) -> dict[str, dict[str, float]]:
        """idf = log10(N / df), 0 if the term occurs in no document (TermFrequency_InverseDocumentFrequency)."""
        n = len(texts)
        df = {"unigrams": Counter(), "bigrams": Counter(), "wordnet": Counter()}
        for text in texts:
            tokens = java_split(text)
            df["unigrams"].update(set(ngrams(tokens, 1)))
            df["bigrams"].update(set(ngrams(tokens, 2)))
            df["wordnet"].update(set(java_split(replace_with_marks(text, self.lists, self.sets))))
        terms = {"unigrams": [t for t, _ in self.unigrams], "bigrams": [t for t, _ in self.bigrams], "wordnet": self.categories}
        self.idf = {k: {t: (math.log10(n / df[k][t]) if df[k][t] else 0.0) for t in terms[k]} for k in terms}
        return self.idf

    def transform(self, texts: list[str]):
        """Sparse matrix (documents × columns) of tf · idf."""
        import numpy as np
        from scipy.sparse import csr_matrix

        index = {}
        for k, terms in (("unigrams", [t for t, _ in self.unigrams]), ("bigrams", [t for t, _ in self.bigrams]),
                         ("wordnet", self.categories)):  # fmt: skip
            for t in terms:
                index[(k, t)] = len(index)
        rows, cols, vals = [], [], []
        for r, text in enumerate(texts):
            tokens = java_split(text)
            for k, grams in (("unigrams", ngrams(tokens, 1)), ("bigrams", ngrams(tokens, 2)),
                             ("wordnet", java_split(replace_with_marks(text, self.lists, self.sets)))):  # fmt: skip
                for g, f in tf(grams).items():
                    j = index.get((k, g))
                    if j is not None and self.idf[k][g]:
                        rows.append(r)
                        cols.append(j)
                        vals.append(f * self.idf[k][g])
        return csr_matrix((np.array(vals), (rows, cols)), shape=(len(texts), len(index)))


def java_preprocess(texts: list[str], emotxt_dir: Path, heap: str = "2000m") -> list[str]:
    """EmoTxt's own Java tokenizer and cleaning (one JVM, `createDocFormat` mode), as for its training corpus."""
    emotxt_dir = Path(emotxt_dir).resolve()
    work = Path(tempfile.mkdtemp(prefix="emotxt_", dir=emotxt_dir))
    out = emotxt_dir / f"classification_{work.name}_anger"
    try:
        src = work / f"{work.name}.csv"
        with open(src, "w", encoding="utf-8") as f:
            f.write("id;label;text\n")
            for i, t in enumerate(texts):
                t = " ".join(t.split()).replace('"', "'")  # one line, no quote characters (as the training corpus)
                f.write(f'{i};NO;"""{t}"""\n')
        subprocess.run(["java", f"-Xmx{heap}", "-jar", str(emotxt_dir / "java" / "Emotion_and_Polarity_SO.jar"),
                        "-i", str(src), "-d", ";", "-t", "classification", "-Ex", "createDocFormat", "-e", "anger"],
                       cwd=emotxt_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # fmt: skip
        lines = (out / "ElaboratedFiles" / "onlyText_PreProcessed.txt").read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if len(lines) != len(texts):  # positional alignment: any mismatch would shift every later label
            raise RuntimeError(f"EmoTxt pre-processing returned {len(lines)} lines for {len(texts)} texts")
        return lines
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
