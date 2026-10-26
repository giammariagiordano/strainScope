"""Java-compatible tokenization details of the EmoTxt feature reimplementation (app/tools/emotxt_features.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "tools"))
from emotxt_features import java_split, ngrams, replace_with_marks, tf  # noqa: E402


def test_split_behaves_like_java():
    assert java_split(" a b  c ") == ["", "a", "b", "c"]  # leading empty kept, trailing dropped
    assert java_split("") == [""]


def test_ngrams_and_term_frequency():
    assert ngrams(["a", "b", "c"], 2) == ["a_b", "b_c"]
    assert tf(["a", "a", "b", ""]) == {"a": 0.5, "b": 0.25, "": 0.25}  # the empty token counts, as in Java


def test_wordnet_marks_first_match_by_file_order():
    lists = {
        "neutral_emotion.csv": {"apathy": ["meh"]},
        "ambiguos-emotion.csv": {"surprise": ["wow"]},
        "positive_emotion.csv": {"joy": ["glad", "wow"]},
        "negative_emotion.csv": {"sadness": ["sad"]},
    }
    assert replace_with_marks("wow so glad not sad joy", lists) == " surprise joy sadness joy"
