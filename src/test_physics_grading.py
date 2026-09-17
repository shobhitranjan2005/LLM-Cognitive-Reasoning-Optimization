"""Exercise the physics grader before it costs GPU time.

An earlier sweep shipped a grader that read the '2' in 'm/s^2' as an answer, and
2 is the correct answer to two of the held-out problems -- so wrong answers
scored correct and nothing looked amiss. These asserts make that class of
mistake fail here instead.

    python test_physics_grading.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_worker import answer_span, closes_think, numbers_in, score  # noqa: E402

FAIL = []


def eq(got, want, what):
    if got != want:
        FAIL.append(f"{what}: got {got!r}, want {want!r}")


# --- the bug that started all this -------------------------------------------
eq(numbers_in("the deceleration is 10.67 m/s^2"), [10.67],
   "a unit exponent is not a number")
eq(numbers_in("g = 9.8 m s^-2 downward"), [9.8], "negative unit exponent")
eq(numbers_in("area 43.7 cm3"), [43.7], "exponent written without a caret")

# --- scientific notation ------------------------------------------------------
eq(numbers_in("about 2.2 x 10^6 m/s"), [2200000.0], "x 10^n")
eq(numbers_in("5.3 × 10 ^ -11 m"), [5.3e-11], "unicode times, negative exponent")
eq(numbers_in("1,250 metres"), [1250.0], "thousands separator")

# --- multi-part scoring -------------------------------------------------------
two = [{"value": 6.90, "tol": 0.02}, {"value": 2.12, "tol": 0.02}]
ok, hits = score("<think>x</think>\nSpeed 6.90 km/s and period 2.12 hours.", two)
eq((ok, hits), (True, [True, True]), "both parts present")
ok, hits = score("<think>x</think>\nSpeed is 6.90 km/s.", two)
eq((ok, hits), (False, [True, False]), "one part missing is not correct")

one = [{"value": 0.2, "tol": 0.02}]
eq(score("<think>x</think>\nThe distance is 0.2 m.", one)[0], True, "single part")
eq(score("<think>x</think>\nThe distance is 0.25 m.", one)[0], False,
   "outside tolerance")
eq(score("<think>x</think>\nThe distance is 0.2016 m.", one)[0], True,
   "inside tolerance")

# --- where the answer is read from -------------------------------------------
# The scratch work mentions 99; only the text after the tag should be scored.
eq(score("<think>\ntry 99 first\n</think>\nThe answer is 0.2 m.", one)[0], True,
   "answer read after the think block")
eq(score("<think>\nthe value is 0.2\n</think>\nSo it is 99 m.", one)[0], False,
   "a right number buried in scratch work does not count")

# The base never closes the block. Falling back to the whole text is the only
# way to score it at all, and it must not crash.
eq(score("<think>\nworking it out, I get 0.2 m", one)[0], True,
   "unclosed block falls back to the whole text")
eq(closes_think("<think>a</think>b"), True, "closing tag detected")
eq(closes_think("<think>a"), False, "no closing tag")

# A closing tag with no number after it: widen rather than score None.
eq(score("<think>\nso 0.2 m\n</think>\nDone.", one)[0], True,
   "no number after the tag falls back")

eq(answer_span("<think>a</think>  final"), "  final", "span is what follows the tag")

if FAIL:
    print(f"{len(FAIL)} FAILURE(S):")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("all physics grading checks passed")
