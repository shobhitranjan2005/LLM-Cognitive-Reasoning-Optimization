"""Local test of the pure logic inside loom_v2_train.py -- no GPU, no 7B model.

Pulls the real function bodies out of the training script by AST (so this tests
the shipped code, not a copy), binds the real DeepSeek tokenizer from the v1
archive, and asserts the grading behaves on cases that actually matter:
scientific notation, thousands separators, multi-part answers, wrong answers,
missing think tags, and regurgitated training text.

    python test_grading_logic.py
"""
import ast
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = (HERE / "loom_v2_train.py").read_text(encoding="utf-8")
TOKENIZER_DIR = ROOT / "models" / "v1_original_2026-08-07" / "adapter"

# ---- lift the functions out of the training script --------------------------
WANT = ["split_heldout", "numbers_in", "answer_span", "is_correct",
        "think_token_count", "_words", "build_corpus_index", "max_shared_ngram",
        "build_example", "collate"]

tree = ast.parse(SRC)
sources = {n.name: ast.get_source_segment(SRC, n)
           for n in tree.body if isinstance(n, ast.FunctionDef)}
missing = [w for w in WANT if w not in sources]
assert not missing, f"functions vanished from the training script: {missing}"

from transformers import AutoTokenizer
import torch

print("loading the real tokenizer from the v1 archive ...")
tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

NS = {"re": re, "math": math, "json": json, "torch": torch, "tok": tok,
      "MAX_LEN": 768, "_SCI": None, "_NUM": None, "_NGRAM_SIZES": None}
# the module-level regexes the functions close over
NS["_SCI"] = re.compile(r"(\d+(?:\.\d+)?)\s*[x*×]\s*10\s*\^?\s*\(?(-?\d+)\)?")
NS["_POW"] = re.compile(r"(\d+(?:\.\d+)?)\s*\^\s*\(?(-?\d+)\)?")
NS["_UNIT_EXP"] = re.compile(r"(?<=[A-Za-z)])\s*\^\s*\(?-?\d+\)?")
NS["_NUM"] = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
NS["_NGRAM_SIZES"] = [5, 8, 12, 20, 30, 45, 60]
for name in WANT:
    exec(sources[name], NS)
NS["TRAIN_NGRAMS"] = None  # set below

numbers_in = NS["numbers_in"]
answer_span = NS["answer_span"]
is_correct = NS["is_correct"]
think_token_count = NS["think_token_count"]
max_shared_ngram = NS["max_shared_ngram"]
build_corpus_index = NS["build_corpus_index"]
split_heldout = NS["split_heldout"]
build_example = NS["build_example"]
collate = NS["collate"]

FAILS = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILS.append(f"{label}: got {got!r}, wanted {want!r}")
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")


print("\n--- numbers_in ------------------------------------------------------")
check("plain", numbers_in("the answer is 20 seconds"), [20.0])
check("thousands separator", numbers_in("6,150 meters"), [6150.0])
check("scientific x10^", numbers_in("travels at 5 x 10^5 m/s"), [500000.0])
check("negative exponent", numbers_in("for 4 x 10^-6 seconds"), [4e-06])
check("decimal, unit exponent stripped", numbers_in("about 10.67 m/s^2"), [10.67])
check("e-notation, unit exponent stripped", numbers_in("9e11 m/s^2"), [9e11])
check("v^2 form does not leak a 2", numbers_in("v^2 = u^2 + 2as gives 32 m/s"), [2.0, 32.0])
check("bare power still evaluated", numbers_in("area is 4^2 units"), [16.0])
check("m/s^2 cannot fake the answer 2",
      is_correct("<think>x</think> the answer is 7 m/s^2", [("t", 2.0, "s", 0.02)]), False)
check("no false positive on words", numbers_in("no digits here"), [])

print("\n--- answer_span -----------------------------------------------------")
check("splits on last </think>",
      answer_span("<think>junk 999</think>\n\nthe answer is 42").strip(),
      "the answer is 42")
check("falls back when tag missing", answer_span("bare 42 text"), "bare 42 text")

print("\n--- is_correct ------------------------------------------------------")
KEY = json.loads((ROOT / "data" / "heldout_answer_key.json").read_text(encoding="utf-8"))
BY_ID = {a["id"]: [(p["name"], p["value"], p["unit"], p["tol"]) for p in a["parts"]]
         for a in KEY["answers"]}

check("single part right",
      is_correct("<think>x</think> the robot draws level after 2 seconds", BY_ID["1D-01-c"]),
      True)
check("single part wrong",
      is_correct("<think>x</think> the answer is 3 seconds", BY_ID["1D-01-c"]),
      False)
check("ignores numbers inside think",
      is_correct("<think>maybe 2 seconds</think> so the answer is 7 seconds", BY_ID["1D-01-c"]),
      False)
check("multi-part both present",
      is_correct("<think>x</think> a = 5 m/s^2 and the tension is 0.01 N", BY_ID["FO-01-c"]),
      True)
check("multi-part one missing",
      is_correct("<think>x</think> the acceleration is 5 m/s^2", BY_ID["FO-01-c"]),
      False)
check("big number with comma",
      is_correct("<think>x</think> the coupling carries 120,000 N at 4 m/s^2", BY_ID["FO-05-c"]),
      True)
check("rounding inside tolerance (16.3 vs 16.26)",
      is_correct("<think>x</think> about 16.3 degrees", BY_ID["2D-03-c"][:1] and
                 [("angle", 16.2602, "deg", 0.02)]),
      True)
check("sqrt form 70.7 accepted",
      is_correct("<think>x</think> v = sqrt(5000), about 70.7 m/s", BY_ID["CM-03-c"]),
      True)
check("tiny value 0.01 not matched by 0.1",
      is_correct("<think>x</think> a = 5 m/s^2, tension 0.1 N", BY_ID["FO-01-c"]),
      False)

print("\n--- think_token_count -----------------------------------------------")
n = think_token_count("<think>\nshort reasoning here\n</think>\n\nanswer")
check("counts a think block", isinstance(n, int) and n > 0, True)
check("no think tag at all -> 0", think_token_count("just an answer"), 0)
check("unclosed think -> None", think_token_count("<think>never closes"), None)

print("\n--- regurgitation ---------------------------------------------------")
ROWS = [json.loads(l) for l in (ROOT / "train.jsonl").open(encoding="utf-8")]
TRAIN_ROWS = [r for r in ROWS if r["split"] == "train"]
HELD = [r for r in ROWS if r["split"] == "test"]
NS["TRAIN_NGRAMS"] = build_corpus_index(TRAIN_ROWS)
idx = NS["TRAIN_NGRAMS"]

verbatim = TRAIN_ROWS[0]["think"]
check("verbatim training text scores maximum", max_shared_ngram(verbatim, idx), 60)
check("unrelated prose scores 0",
      max_shared_ngram("The weather today is mild and the harbour is quiet.", idx), 0)
held_out_text = HELD[0]["think"]
ng_held = max_shared_ngram(held_out_text, idx)
print(f"       (a held-out trace shares {ng_held} words with training -- "
      f"expected small, it is unseen text)")
check("held-out trace is not flagged as regurgitation", ng_held < 20, True)

print("\n--- split_heldout ---------------------------------------------------")
val, pristine = split_heldout(HELD)
check("10 val", len(val), 10)
check("10 pristine", len(pristine), 10)
check("no overlap", set(r["id"] for r in val) & set(r["id"] for r in pristine), set())
check("covers all 20", len(set(r["id"] for r in val + pristine)), 20)
chapters_val = sorted({r["chapter"] for r in val})
check("val spans all four chapters", len(chapters_val), 4)
val2, pristine2 = split_heldout(list(reversed(HELD)))
check("deterministic under input order",
      [r["id"] for r in val] == [r["id"] for r in val2], True)
print("       val      :", [r["id"] for r in val])
print("       pristine :", [r["id"] for r in pristine])

print("\n--- prompt build + label masking ------------------------------------")
NS["tok"] = tok
ex = build_example(TRAIN_ROWS[0])
ids, labels = ex["input_ids"], ex["labels"]
check("same length", len(ids) == len(labels), True)
check("prompt is masked", labels[0], -100)
n_sup = sum(1 for x in labels if x != -100)
check("response is supervised", n_sup > 50, True)
supervised = tok.decode([x for x in labels if x != -100])
check("supervised part starts with <think>", supervised.lstrip().startswith("<think>"), True)
check("supervised part ends at eos", labels[-1] == tok.eos_token_id, True)
lens = [len(build_example(r)["input_ids"]) for r in ROWS]
check("nothing truncated at MAX_LEN=768", max(lens) < 768, True)
print(f"       token lengths: min {min(lens)} mean {sum(lens)//len(lens)} max {max(lens)}")

batch = collate([build_example(r) for r in TRAIN_ROWS[:3]])
check("collate pads to a rectangle",
      batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape, True)
check("padding is masked out of the loss",
      all(batch["labels"][i][batch["attention_mask"][i] == 0].eq(-100).all()
          for i in range(3)), True)

print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("all grading logic passed -- safe to run on the T4")
