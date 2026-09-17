"""Exercise the GSM8K answer parsers before they cost GPU time.

The last sweep shipped a grader that read the '2' in 'm/s^2' as an answer, and
2 was the right answer to two held-out problems -- so wrong answers scored
correct and nobody would have noticed. These asserts exist so that class of
mistake has to fail here instead.

    python test_gsm8k_grading.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_worker import gold_answer, predicted_answer  # noqa: E402

FAIL = []


def eq(got, want, what):
    if got != want:
        FAIL.append(f"{what}: got {got!r}, want {want!r}")


# --- gold answers, exactly as GSM8K writes them ------------------------------
eq(gold_answer("Janet sells the rest for $2 each.\n#### 18"), 18.0, "plain gold")
eq(gold_answer("...\n#### 1,250"), 1250.0, "gold with a thousands separator")
eq(gold_answer("...\n#### 5.5"), 5.5, "decimal gold")
eq(gold_answer("...\n#### -3"), -3.0, "negative gold")
eq(gold_answer("no marker here"), None, "missing gold marker")

# --- the model's answer ------------------------------------------------------
eq(predicted_answer("<think>\n3+4=7, then 7*2=14\n</think>\nThe answer is 28."),
   28.0, "answer after the think block, not the scratch work inside it")
eq(predicted_answer("<think>\n72/2 = 36\n</think>\n36"), 36.0, "bare final number")
eq(predicted_answer("<think>x</think>\nShe earns $18 per day."), 18.0,
   "currency symbol stripped")
eq(predicted_answer("<think>y</think>\nTotal: 1,250 dollars"), 1250.0,
   "thousands separator in the prediction")
eq(predicted_answer("<think>z</think>\nThe cost is 12.50."), 12.50,
   "trailing full stop is not part of the number")

# The block never closed: the budget ran out mid-trace. Falling back to the
# whole text is the honest reading, and it must not crash or return None.
eq(predicted_answer("<think>\nfirst 5, then 10, so 15"), 15.0,
   "unclosed think block falls back to the full text")

# A closing tag with nothing numeric after it. Scoring the empty tail would
# throw the answer away, so the search widens to the whole text.
eq(predicted_answer("<think>\n6*7 = 42\n</think>\nDone."), 42.0,
   "no number after the tag, so fall back rather than score None")

eq(predicted_answer("<think>a</think>\nno digits at all"), None,
   "genuinely no number anywhere")

# Ordering: the LAST number after the tag is the answer, not the first.
eq(predicted_answer("<think>q</think>\nWe had 10 and 4, so the total is 14."),
   14.0, "last number wins")

if FAIL:
    print(f"{len(FAIL)} FAILURE(S):")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("all GSM8K grading checks passed")
