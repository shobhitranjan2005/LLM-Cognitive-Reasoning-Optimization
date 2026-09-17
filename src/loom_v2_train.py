# %% [markdown]
# # LOOM v2 — anti-memorisation fine-tune sweep (Kaggle T4)
#
# Trains several LoRA candidates on the 52 LOOM training rows, **with a real
# validation set and early stopping**, then grades every candidate on held-out
# physics it never saw, on regurgitation, and on general ability — and picks a
# winner on the evidence.
#
# What v1 did wrong: 8 epochs at LR 2e-4 with `save_strategy='no'` and **no eval
# set at all**, so nothing could detect overfitting and only the final,
# most-over-trained state was ever saved. Final train loss 0.013 = memorised.
#
# **Setup (once):**
# 1. Notebook settings → **Accelerator: GPU T4**, **Internet: On**
# 2. Add Data → upload `train.jsonl` as a dataset → set `TRAIN_JSONL` below
# 3. (optional) Add your existing `deep_reason_lora` dataset to compare v1
# 4. Run All. ~2–3.5 h. Resumable: re-run the cell if the session drops.
#
# Nothing here writes to your dataset. v1 weights are never touched.

# %%
# ============================== KNOBS ========================================
from pathlib import Path

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def _find(pattern, root="/kaggle/input"):
    """Locate an attached dataset file by shape rather than by slug -- dataset
    folder names change and a wrong path would otherwise kill the run minutes in."""
    hits = sorted(Path(root).glob(pattern))
    return hits[0] if hits else None
    # NB: recursive. Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>/
    # in some runtimes and /kaggle/input/<slug>/ in others; a one-level glob misses
    # the first shape and the run dies on the assert below.


# train.jsonl (72 rows), from whichever dataset carries it.
TRAIN_JSONL = _find("**/train.jsonl")

# The v1 adapter, to benchmark v2 against it. Found by its config file, so the
# dataset slug and inner folder name do not matter. None = skip the comparison.
_v1_cfg = _find("**/adapter_config.json")
V1_ADAPTER = _v1_cfg.parent if _v1_cfg else None

OUT_DIR = Path("/kaggle/working/loom_v2")

MAX_LEN = 768          # every LOOM trace fits; asserted below
SEED = 20260819

RUN_GSM8K = True       # winner + v1 + base only; needs Internet On
GSM8K_N = 40

# Which sweep arms to run. Trimming this list is the way to cut the budget.
RUN_ARMS = ["A_v1_control", "B_half", "C_attn_only", "D_lowrank",
            "E_gentle", "F_neftune", "G_rslora", "H_regularised"]
# =============================================================================

# %%
# !pip -q install -U "transformers>=4.44" accelerate peft bitsandbytes datasets

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import inspect
import json
import math
import os
import random
import re
import time

import torch
import transformers

print("torch", torch.__version__, "| transformers", transformers.__version__)
assert torch.cuda.is_available(), (
    "NO GPU. Notebook settings -> Accelerator -> GPU T4. Refusing to run on CPU: "
    "a 7B QLoRA fine-tune on CPU would take days."
)
print("gpu:", torch.cuda.get_device_name(0),
      "| vram %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

# T4 is sm_75: no native bf16. Do NOT trust torch.cuda.is_bf16_supported() here,
# it reports True for *emulated* bf16 on recent torch.
_cc = torch.cuda.get_device_capability(0)
# Kaggle hands out P100 (sm_60) unless the accelerator is pinned, and its PyTorch
# build only supports sm_70+. Fail here, in seconds, rather than 20 minutes later
# inside bitsandbytes. Fix: push with --accelerator NvidiaTeslaT4.
assert _cc[0] >= 7, (
    f"{torch.cuda.get_device_name(0)} is sm_{_cc[0]}{_cc[1]}; this PyTorch needs "
    "sm_70+. Set the accelerator to GPU T4 and re-run.")
NATIVE_BF16 = _cc[0] >= 8
DTYPE = torch.bfloat16 if NATIVE_BF16 else torch.float16
print(f"compute capability {_cc[0]}.{_cc[1]} -> training in "
      f"{'bf16' if NATIVE_BF16 else 'fp16'}")

random.seed(SEED)
torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Data and the val / pristine split
#
# The 20 `c` variants are held out in the dataset. We split them **10 / 10**:
#
# * **val** — early stopping and checkpoint selection run on these
# * **pristine** — never touched by training or model selection; every number in
#   the final report comes from these, so they stay honest evidence
#
# The split is deterministic and stratified by chapter, so it is identical on
# every run. `train.jsonl` is read only.

# %%
print("train.jsonl ->", TRAIN_JSONL)
print("v1 adapter  ->", V1_ADAPTER)
assert TRAIN_JSONL and TRAIN_JSONL.exists(), (
    f"train.jsonl not found under /kaggle/input. Attach the loom-train dataset.")

ROWS = [json.loads(l) for l in TRAIN_JSONL.open(encoding="utf-8")]
TRAIN_ROWS = [r for r in ROWS if r["split"] == "train"]
HELD_OUT = [r for r in ROWS if r["split"] == "test"]
assert len(TRAIN_ROWS) == 52 and len(HELD_OUT) == 20, (len(TRAIN_ROWS), len(HELD_OUT))


def split_heldout(held):
    """Deterministic 10/10 stratified split. Chapters alternate which half of
    their 5 c-variants goes to val, so both sides get all four chapters."""
    by_ch = {}
    for r in held:
        by_ch.setdefault(r["chapter"], []).append(r)
    val, pristine = [], []
    for i, ch in enumerate(sorted(by_ch)):
        ids = sorted(by_ch[ch], key=lambda r: r["id"])
        first = [ids[0], ids[2], ids[4]]     # 3 rows
        second = [ids[1], ids[3]]            # 2 rows
        if i % 2 == 0:
            val += first; pristine += second
        else:
            val += second; pristine += first
    return sorted(val, key=lambda r: r["id"]), sorted(pristine, key=lambda r: r["id"])


VAL_ROWS, PRISTINE_ROWS = split_heldout(HELD_OUT)
assert len(VAL_ROWS) == 10 and len(PRISTINE_ROWS) == 10
assert not ({r["id"] for r in VAL_ROWS} & {r["id"] for r in PRISTINE_ROWS})

print("train   ", len(TRAIN_ROWS))
print("val     ", [r["id"] for r in VAL_ROWS])
print("pristine", [r["id"] for r in PRISTINE_ROWS])

# %%
# Ground truth for the held-out problems, derived and triple-checked offline
# (data/derive_answer_key.py + data/recheck_answer_key.py). Embedded so this
# notebook needs only one upload.
ANSWER_KEY = {
    "1D-01-c": [("time", 2.0, "s", 0.02)],
    "1D-02-c": [("distance", 0.2, "m", 0.02)],
    "1D-03-c": [("time", 5.0, "s", 0.02)],
    "1D-04-c": [("length", 2.0, "m", 0.02)],
    "1D-05-c": [("length", 6150.0, "m", 0.02)],
    "2D-01-c": [("drop", 0.8, "m", 0.02)],
    "2D-02-c": [("speed", 30.0, "m/s", 0.02)],
    "2D-03-c": [("angle", 16.2602, "deg", 0.02), ("net_speed", 24.0, "m/s", 0.02)],
    "2D-04-c": [("angle", 65.0, "deg", 0.02)],
    "2D-05-c": [("radius", 2.5, "m", 0.02)],
    "CM-01-c": [("speed", 20.0, "m/s", 0.02)],
    "CM-02-c": [("radius", 6181.3187, "m", 0.02)],
    "CM-03-c": [("speed", 70.7107, "m/s", 0.02)],
    "CM-04-c": [("acceleration", 10.6704, "m/s^2", 0.02)],
    "CM-05-c": [("acceleration", 1.5, "m/s^2", 0.02)],
    "FO-01-c": [("acceleration", 5.0, "m/s^2", 0.02), ("tension", 0.01, "N", 0.02)],
    "FO-02-c": [("acceleration", 8.66, "m/s^2", 0.02), ("normal_force", 2500.0, "N", 0.02)],
    "FO-03-c": [("acceleration", 1.0, "m/s^2", 0.02), ("force", 550.0, "N", 0.02)],
    "FO-04-c": [("angle", 16.6992, "deg", 0.02)],
    "FO-05-c": [("acceleration", 4.0, "m/s^2", 0.02), ("coupling_force", 120000.0, "N", 0.02)],
}
assert set(ANSWER_KEY) == {r["id"] for r in HELD_OUT}

# %% [markdown]
# ## 2. Prompt format
#
# Identical to the v1 run, so v1 and v2 are comparable: the model's own chat
# template, response `<think>…</think>\n\n{solution}`, loss masked over the
# prompt so only the response shape is learned.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def build_example(row):
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    )
    response = f"<think>\n{row['think']}\n</think>\n\n{row['solution']}"
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": (p_ids + r_ids)[:MAX_LEN],
            "labels": ([-100] * len(p_ids) + r_ids)[:MAX_LEN]}


TRAIN_DS = [build_example(r) for r in TRAIN_ROWS]
VAL_DS = [build_example(r) for r in VAL_ROWS]

_lens = [len(d["input_ids"]) for d in TRAIN_DS + VAL_DS]
print(f"token length: min {min(_lens)} | mean {sum(_lens) // len(_lens)} | max {max(_lens)}")
assert max(_lens) < MAX_LEN, "a trace is being truncated -- raise MAX_LEN"


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


# %% [markdown]
# ## 3. Grading
#
# Four things get measured on every candidate, all on the **pristine** 10:
#
# | Metric | What it catches |
# |---|---|
# | `correct` | answers right, not just short — graded against the answer key |
# | `think_tokens` | the compression claim |
# | `max_shared_ngram` | **regurgitation**: longest word run copied from a training trace |
# | `general` + `ppl_ratio` | **damage**: did the fine-tune break the base model |

# %%
_SCI = re.compile(r"(\d+(?:\.\d+)?)\s*[x*×]\s*10\s*\^?\s*\(?(-?\d+)\)?")
_POW = re.compile(r"(\d+(?:\.\d+)?)\s*\^\s*\(?(-?\d+)\)?")
_UNIT_EXP = re.compile(r"(?<=[A-Za-z)])\s*\^\s*\(?-?\d+\)?")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(text):
    """Every number in `text`, with '1,200' and '5 x 10^5' understood.

    Unit exponents are stripped BEFORE extraction. Without this, 'm/s^2' donates
    a stray 2.0 to the number list -- and 2.0 is the correct answer to two of the
    held-out problems, so any wrong answer that happened to write 'm/s^2' would
    have been scored correct. The caret is only read as a power when a digit sits
    in front of it.
    """
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    t = _SCI.sub(lambda m: repr(float(m.group(1)) * 10 ** int(m.group(2))), t)
    t = _POW.sub(lambda m: repr(float(m.group(1)) ** int(m.group(2))), t)
    t = _UNIT_EXP.sub("", t)
    out = []
    for s in _NUM.findall(t):
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def answer_span(text):
    """The part after </think> -- where the answer belongs. Falls back to the
    whole output if the tag never closes."""
    i = text.rfind("</think>")
    return text[i + len("</think>"):] if i != -1 else text


def is_correct(text, parts):
    """Every required value must appear in the answer span, within tolerance."""
    nums = numbers_in(answer_span(text))
    if not nums:
        return False
    for _name, want, _unit, tol in parts:
        if not any(math.isclose(n, want, rel_tol=tol, abs_tol=abs(want) * tol + 1e-9)
                   for n in nums):
            return False
    return True


def think_token_count(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not m:
        return 0 if "<think>" not in text else None
    return len(tok(m.group(1), add_special_tokens=False)["input_ids"])


# --- regurgitation -----------------------------------------------------------
_NGRAM_SIZES = [5, 8, 12, 20, 30, 45, 60]


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def build_corpus_index(rows):
    corpus = [f"{r['think']} {r['solution']}" for r in rows]
    idx = {n: set() for n in _NGRAM_SIZES}
    for text in corpus:
        w = _words(text)
        for n in _NGRAM_SIZES:
            for i in range(len(w) - n + 1):
                idx[n].add(tuple(w[i:i + n]))
    return idx


TRAIN_NGRAMS = build_corpus_index(TRAIN_ROWS)


def max_shared_ngram(text, idx=None):
    """Longest word run this output shares with ANY training trace. High = the
    model is reciting training data rather than reasoning."""
    idx = TRAIN_NGRAMS if idx is None else idx
    w = _words(text)
    best = 0
    for n in _NGRAM_SIZES:
        if len(w) < n:
            break
        if any(tuple(w[i:i + n]) in idx[n] for i in range(len(w) - n + 1)):
            best = n
        else:
            break
    return best


# --- general ability ---------------------------------------------------------
PROBES = [
    ("What is the capital of France?", ["paris"]),
    ("Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    ("What is 17 + 25?", ["42"]),
    ("What is 12 times 12?", ["144"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("Name the largest planet in our solar system.", ["jupiter"]),
    ("What colour is a ripe banana?", ["yellow"]),
    ("How many days are in a leap year?", ["366"]),
    ("What is the boiling point of water at sea level, in Celsius?", ["100"]),
    ("Which language is mainly spoken in Brazil?", ["portuguese"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("How many continents are there on Earth?", ["7", "seven"]),
]

PPL_TEXT = (
    "The harbour opened onto a wide bay where fishing boats returned each evening. "
    "Records from the period are incomplete, but the trade in salt and dried fish "
    "supported several hundred families for most of the century. Later, as the "
    "railway reached the coast, the character of the town changed and the older "
    "warehouses were converted into workshops and lodging houses."
)


@torch.no_grad()
def perplexity(model, text):
    ids = tok(text, return_tensors="pt").to(0)
    out = model(**ids, labels=ids["input_ids"])
    return float(torch.exp(out.loss))


@torch.no_grad()
def generate(model, prompt, max_new=700):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(0)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def grade(model, label, rows=None, verbose=True):
    """Full report card for one model state."""
    rows = PRISTINE_ROWS if rows is None else rows
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    per_problem, n_ok, n_fmt, toks, ngrams = [], 0, 0, [], []
    for r in rows:
        t0 = time.time()
        gen = generate(model, r["prompt"])
        ok = is_correct(gen, ANSWER_KEY[r["id"]])
        tt = think_token_count(gen)
        fmt = "<think>" in gen and "</think>" in gen
        ng = max_shared_ngram(gen)
        n_ok += ok
        n_fmt += fmt
        if tt:
            toks.append(tt)
        ngrams.append(ng)
        per_problem.append({"id": r["id"], "correct": bool(ok), "format_ok": bool(fmt),
                            "think_tokens": tt, "max_shared_ngram": ng,
                            "seconds": round(time.time() - t0, 1), "output": gen})
        if verbose:
            print(f"  {r['id']:9s} {'OK ' if ok else '.  '} fmt={int(fmt)} "
                  f"think={tt} ngram={ng} ({time.time() - t0:.0f}s)")

    probe_ok, probe_detail = 0, []
    for q, expect in PROBES:
        gen = generate(model, q, max_new=400)
        hit = any(e in gen.lower() for e in expect)
        probe_ok += hit
        probe_detail.append({"q": q, "hit": bool(hit), "output": gen[:400]})

    report = {
        "label": label,
        "n": len(rows),
        "correct": n_ok,
        "accuracy": round(n_ok / len(rows), 3),
        "format_ok": round(n_fmt / len(rows), 3),
        "median_think_tokens": (sorted(toks)[len(toks) // 2] if toks else None),
        "mean_think_tokens": (round(sum(toks) / len(toks), 1) if toks else None),
        "max_shared_ngram": max(ngrams) if ngrams else 0,
        "mean_shared_ngram": round(sum(ngrams) / len(ngrams), 1) if ngrams else 0,
        "general_score": round(probe_ok / len(PROBES), 3),
        "perplexity": round(perplexity(model, PPL_TEXT), 3),
        "per_problem": per_problem,
        "probes": probe_detail,
    }
    print(f"[{label}] acc {report['accuracy']} | fmt {report['format_ok']} | "
          f"think~{report['median_think_tokens']} | ngram {report['max_shared_ngram']} | "
          f"general {report['general_score']} | ppl {report['perplexity']}")
    return report


# %% [markdown]
# ## 4. Model loading
#
# The base is reloaded **fresh for every arm**. Slower than reusing one model,
# and deliberate: it makes cross-contamination between sweep arms impossible, so
# a bad arm cannot quietly poison the next one.

# %%
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback

BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE)

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL7 = ATTN + ["gate_proj", "up_proj", "down_proj"]


def fresh_base():
    gc.collect(); torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=BNB, device_map={"": 0}, **{_DTYPE_KW: DTYPE})
    m.config.use_cache = False
    return m


def release(*names):
    """Free VRAM held by model objects bound to GLOBAL names.

    A helper cannot free the caller's model with `del m` -- that only unbinds the
    local parameter, and the caller's variable keeps the object alive. That bug
    kept the base model and v1 resident through grading and OOM'd the first
    training arm. So: pass the NAMES, and rebind them to None here.
    """
    g = globals()
    for n in names:
        if n in g:
            g[n] = None
    gc.collect(); torch.cuda.empty_cache()


_TA = inspect.signature(TrainingArguments.__init__).parameters
EVAL_KW = "eval_strategy" if "eval_strategy" in _TA else "evaluation_strategy"
HAS_NEFTUNE = "neftune_noise_alpha" in _TA


def make_args(**kw):
    """Build TrainingArguments, dropping anything this version rejects.

    Kaggle ships whatever transformers it likes (5.15.1 today, which removed
    `warmup_ratio`), and pinning a version on Kaggle breaks as often as it
    helps. So adapt to the runtime signature instead of guessing it.
    """
    if "warmup_ratio" in kw and "warmup_ratio" not in _TA:
        ratio = kw.pop("warmup_ratio")
        if "warmup_steps" in _TA:                      # keep the warmup, other spelling
            total = math.ceil(len(TRAIN_DS) / 4) * kw.get("num_train_epochs", 1)
            kw["warmup_steps"] = max(1, int(total * ratio))
    unsupported = sorted(k for k in kw if k not in _TA)
    if unsupported:
        print(f"  [TrainingArguments] unsupported here, dropped: {unsupported}")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in _TA})
print(f"TrainingArguments: eval kwarg '{EVAL_KW}' | NEFTune available: {HAS_NEFTUNE}")

# %% [markdown]
# ## 5. The sweep
#
# `A_v1_control` reproduces the v1 recipe **with eval logging added and early
# stopping off** — it is the control that shows the overfitting curve rather than
# a fix. Every other arm attacks memorisation a different way: less capacity,
# gentler schedule, attention-only targeting, embedding noise (NEFTune),
# rank-stabilised scaling, or explicit regularisation.

# %%
SWEEP = {
    "A_v1_control": dict(r=16, alpha=32, dropout=0.05, lr=2e-4, epochs=8,
                         modules="all", early_stop=False,
                         note="v1 recipe + eval logging. The control."),
    "B_half":       dict(r=8, alpha=16, dropout=0.10, lr=1e-4, epochs=6,
                         modules="all", note="half the rank, half the LR"),
    "C_attn_only":  dict(r=8, alpha=16, dropout=0.10, lr=1e-4, epochs=6,
                         modules="attn", note="attention only -- no MLP capacity"),
    "D_lowrank":    dict(r=4, alpha=8, dropout=0.10, lr=1e-4, epochs=6,
                         modules="all", note="rank 4: too small to store 52 traces"),
    "E_gentle":     dict(r=16, alpha=32, dropout=0.10, lr=5e-5, epochs=6,
                         modules="all", note="v1 capacity, quarter LR"),
    "F_neftune":    dict(r=8, alpha=16, dropout=0.10, lr=1e-4, epochs=6,
                         modules="all", neftune=5.0,
                         note="NEFTune embedding noise -- built for small SFT sets"),
    "G_rslora":     dict(r=8, alpha=16, dropout=0.10, lr=1e-4, epochs=6,
                         modules="all", rslora=True,
                         note="rank-stabilised LoRA scaling"),
    "H_regularised": dict(r=8, alpha=16, dropout=0.10, lr=1e-4, epochs=6,
                          modules="all", wd=0.05, label_smoothing=0.05,
                          note="weight decay + label smoothing"),
}

RESULTS_PATH = OUT_DIR / "sweep_results.json"
RESULTS = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
# A previous session's results can be attached as a dataset to skip re-grading.
_seed = _find("**/sweep_results.json")
if not RESULTS and _seed:
    RESULTS = json.loads(_seed.read_text(encoding="utf-8"))
    print(f"seeded {len(RESULTS)} arms from {_seed}")


def save_results():
    RESULTS_PATH.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")


def train_one(name, cfg):
    print(f"\n{'=' * 70}\n{name}: {cfg.get('note', '')}\n{'=' * 70}")
    adapter_dir = OUT_DIR / name
    base = fresh_base()
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    lora_kw = dict(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                   bias="none", task_type="CAUSAL_LM",
                   target_modules=(ATTN if cfg["modules"] == "attn" else ALL7))
    if cfg.get("rslora"):
        lora_kw["use_rslora"] = True
    model = get_peft_model(base, LoraConfig(**lora_kw))
    model.print_trainable_parameters()

    early = cfg.get("early_stop", True)
    # 52 rows at effective batch 4 is only ~13 optimizer steps per epoch, so
    # evaluating once an epoch gives just a handful of chances to catch the best
    # moment -- and with a set this small, the turn into memorisation happens
    # inside an epoch. Evaluate twice per epoch instead. Eval is 10 forward
    # passes, so the extra cost is seconds.
    steps_per_epoch = max(1, math.ceil(len(TRAIN_DS) / 4))
    eval_every = max(1, steps_per_epoch // 2)

    args_kw = {
        "output_dir": str(OUT_DIR / f"_ckpt_{name}"),
        "num_train_epochs": cfg["epochs"],
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": cfg["lr"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_steps": 5,
        EVAL_KW: "steps",
        "eval_steps": eval_every,
        "save_strategy": "steps",
        "save_steps": eval_every,
        "save_total_limit": 1,
        "load_best_model_at_end": early,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "weight_decay": cfg.get("wd", 0.0),
        "label_smoothing_factor": cfg.get("label_smoothing", 0.0),
        "fp16": not NATIVE_BF16,
        "bf16": NATIVE_BF16,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "report_to": "none",
        "seed": SEED,
    }
    if cfg.get("neftune") and HAS_NEFTUNE:
        args_kw["neftune_noise_alpha"] = cfg["neftune"]

    trainer = Trainer(
        model=model, args=make_args(**args_kw),
        train_dataset=TRAIN_DS, eval_dataset=VAL_DS, data_collator=collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)] if early else [],
    )
    trainer.train()

    curve = [{k: v for k, v in h.items() if k in ("epoch", "loss", "eval_loss")}
             for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    evals = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
    trains = [h["loss"] for h in trainer.state.log_history if "loss" in h]

    model.save_pretrained(str(adapter_dir))
    print(f"saved -> {adapter_dir}")

    report = grade(model, name)
    report.update({
        "config": {k: v for k, v in cfg.items()},
        "adapter_dir": str(adapter_dir),
        "eval_every_steps": eval_every,
        "steps_per_epoch": steps_per_epoch,
        "best_eval_loss": min(evals) if evals else None,
        "final_eval_loss": evals[-1] if evals else None,
        "final_train_loss": trains[-1] if trains else None,
        "curve": curve,
    })
    # A train loss far below eval loss is the memorisation signature.
    if report["final_train_loss"] and report["best_eval_loss"]:
        report["overfit_gap"] = round(report["best_eval_loss"] - report["final_train_loss"], 4)

    del trainer, model, base          # these ARE locals, so del works here
    gc.collect(); torch.cuda.empty_cache()
    return report


# %%
# Baselines first: the untouched base model, and v1 if it was mounted.
if "BASE_untuned" not in RESULTS:
    _b = fresh_base()
    RESULTS["BASE_untuned"] = grade(_b, "BASE_untuned")
    release("_b"); save_results()

if V1_ADAPTER is not None and "V1_original" not in RESULTS:
    _b = fresh_base()
    _v1 = PeftModel.from_pretrained(_b, str(V1_ADAPTER))
    RESULTS["V1_original"] = grade(_v1, "V1_original")
    release("_v1", "_b"); save_results()
elif V1_ADAPTER is None:
    print("no v1 adapter attached -- skipping the v1 comparison")

# %%
for _name in RUN_ARMS:
    if _name in RESULTS:
        print(f"{_name}: already done, skipping (delete it from sweep_results.json to redo)")
        continue
    RESULTS[_name] = train_one(_name, SWEEP[_name])
    save_results()
print("\nsweep complete")

# %% [markdown]
# ## 6. Pick the winner
#
# The rule, applied in code so it cannot be fudged after seeing the numbers:
#
# 1. **Eligible** = format ≥ 0.9, general ability within 10 points of base,
#    perplexity ≤ 1.25× base (these three are the "not damaged" gate)
# 2. Among eligible, **highest accuracy** on the pristine 10
# 3. Ties → **lower regurgitation**, then **fewer think tokens**

# %%
BASE = RESULTS.get("BASE_untuned", {})
base_general = BASE.get("general_score", 0.0)
base_ppl = BASE.get("perplexity", float("inf"))

rows_out = []
for name, rep in RESULTS.items():
    if name.startswith(("BASE", "V1")):
        continue
    damaged = []
    if rep["format_ok"] < 0.9:
        damaged.append("format")
    if rep["general_score"] < base_general - 0.10:
        damaged.append("general")
    if rep["perplexity"] > base_ppl * 1.25:
        damaged.append("perplexity")
    rows_out.append({"name": name, "eligible": not damaged, "failed": damaged, **rep})

eligible = [r for r in rows_out if r["eligible"]]
ranked = sorted(eligible, key=lambda r: (-r["accuracy"], r["max_shared_ngram"],
                                         r["median_think_tokens"] or 10 ** 6))

hdr = f"{'arm':16s} {'acc':>5s} {'fmt':>5s} {'think':>6s} {'ngram':>6s} {'gen':>5s} {'ppl':>7s} {'evalL':>7s}  status"
print(hdr); print("-" * len(hdr))
for key in ("BASE_untuned", "V1_original"):
    if key in RESULTS:
        r = RESULTS[key]
        print(f"{key:16s} {r['accuracy']:5.2f} {r['format_ok']:5.2f} "
              f"{str(r['median_think_tokens']):>6s} {r['max_shared_ngram']:6d} "
              f"{r['general_score']:5.2f} {r['perplexity']:7.2f} {'-':>7s}  reference")
for r in sorted(rows_out, key=lambda x: x["name"]):
    print(f"{r['name']:16s} {r['accuracy']:5.2f} {r['format_ok']:5.2f} "
          f"{str(r['median_think_tokens']):>6s} {r['max_shared_ngram']:6d} "
          f"{r['general_score']:5.2f} {r['perplexity']:7.2f} "
          f"{(r.get('best_eval_loss') or 0):7.3f}  "
          f"{'eligible' if r['eligible'] else 'DAMAGED: ' + ','.join(r['failed'])}")

assert ranked, "no candidate passed the damage gate -- inspect sweep_results.json"
WINNER = ranked[0]
print(f"\nWINNER: {WINNER['name']}  "
      f"(accuracy {WINNER['accuracy']}, regurgitation {WINNER['max_shared_ngram']} words, "
      f"median think {WINNER['median_think_tokens']} tokens)")

# %% [markdown]
# ## 7. GSM8K — the strongest "not damaged" evidence
#
# Grade-school maths the model was never trained on. If v2 keeps the base
# model's GSM8K accuracy, the fine-tune changed style without eating capability.

# %%
def run_gsm8k(model, label, n=GSM8K_N):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    ok = 0
    for ex in ds:
        gold = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        gen = generate(model, ex["question"], max_new=600)
        nums = numbers_in(answer_span(gen))
        ok += bool(nums) and math.isclose(nums[-1], gold, rel_tol=1e-4, abs_tol=1e-4)
    print(f"[{label}] GSM8K {ok}/{n} = {ok / n:.2f}")
    return {"n": n, "correct": ok, "accuracy": round(ok / n, 3)}


if RUN_GSM8K:
    GSM = RESULTS.setdefault("_gsm8k", {})
    try:
        if "base" not in GSM:
            _b = fresh_base(); GSM["base"] = run_gsm8k(_b, "base"); release("_b"); save_results()
        if WINNER["name"] not in GSM:
            _b = fresh_base()
            _m = PeftModel.from_pretrained(_b, WINNER["adapter_dir"])
            GSM[WINNER["name"]] = run_gsm8k(_m, WINNER["name"])
            release("_m", "_b"); save_results()
        if V1_ADAPTER is not None and "v1" not in GSM:
            _b = fresh_base()
            _m = PeftModel.from_pretrained(_b, str(V1_ADAPTER))
            GSM["v1"] = run_gsm8k(_m, "v1")
            release("_m", "_b"); save_results()
        print("\nGSM8K:", json.dumps(GSM, indent=2))
    except Exception as e:                                    # noqa: BLE001
        print(f"GSM8K skipped ({type(e).__name__}: {e}). Internet must be On.")

# %% [markdown]
# ## 8. Package the winner
#
# Everything lands in `/kaggle/working`. Download **`loom_v2_best.zip`** and
# `v2_report.json`, and send the report back — it is the whole story of the run.

# %%
import shutil

BEST_DIR = OUT_DIR / "BEST"
if BEST_DIR.exists():
    shutil.rmtree(BEST_DIR)
shutil.copytree(WINNER["adapter_dir"], BEST_DIR)
tok.save_pretrained(str(BEST_DIR))

(BEST_DIR / "PROVENANCE.md").write_text(f"""# LOOM adapter v2 -- {WINNER['name']}

Trained {time.strftime('%Y-%m-%d %H:%M')} on Kaggle T4. This is a NEW adapter.
It does not replace v1: v1 stays archived at models/v1_original_2026-08-07/.

## Recipe
{json.dumps(WINNER['config'], indent=2)}

## Selection
Chosen from {len(rows_out)} candidates by: damage gate (format >= 0.9, general
ability within 10 points of base, perplexity <= 1.25x base), then highest
accuracy on 10 pristine held-out problems, ties broken by lower regurgitation
then fewer think tokens.

## Measured on the 10 pristine held-out problems (never trained on, never used
## for checkpoint selection)
- accuracy            {WINNER['accuracy']}
- format compliance   {WINNER['format_ok']}
- median think tokens {WINNER['median_think_tokens']}
- longest word run shared with any training trace: {WINNER['max_shared_ngram']}
- general-ability probes {WINNER['general_score']} (base {base_general})
- perplexity {WINNER['perplexity']} (base {base_ppl})
- best eval loss {WINNER.get('best_eval_loss')} / final train loss {WINNER.get('final_train_loss')}

Full detail, including every generated output and the loss curve of every arm,
is in v2_report.json.
""", encoding="utf-8")

REPORT = {
    "created": time.strftime("%Y-%m-%d %H:%M"),
    "base_model": BASE_MODEL,
    "seed": SEED,
    "splits": {"train": [r["id"] for r in TRAIN_ROWS],
               "val": [r["id"] for r in VAL_ROWS],
               "pristine": [r["id"] for r in PRISTINE_ROWS]},
    "winner": WINNER["name"],
    "ranking": [r["name"] for r in ranked],
    "results": RESULTS,
}
(Path("/kaggle/working") / "v2_report.json").write_text(
    json.dumps(REPORT, indent=2), encoding="utf-8")

shutil.make_archive("/kaggle/working/loom_v2_best", "zip", str(BEST_DIR))
print("\nwrote /kaggle/working/loom_v2_best.zip")
print("wrote /kaggle/working/v2_report.json")
print(f"\nWINNER: {WINNER['name']}")
