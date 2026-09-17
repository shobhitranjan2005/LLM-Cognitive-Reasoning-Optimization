"""Train (optionally) and score one arm on the 150-problem physics bundle.

One process, one GPU, one arm -- the same shape as the earlier workers, and for
the same reason: a process exit is the only reliable way to get every byte of
VRAM back, and one worker per card lets two arms run at once.

Two jobs in one file:

  --train         fine-tune from the base first, then score the result
  --adapter PATH  score an adapter that already exists
  neither         score the untouched base

The scoring half is batched. The previous sweep generated one problem at a time
at ~20 s each; 150 problems x 5 arms that way is over four hours of pure
waiting, and batching is the whole difference.

    python physics_worker.py --arm BASE --bundle b.json --out /kaggle/working/phys
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_LEN = 2048

# ---------------------------------------------------------------- grading

# The times sign and the unicode minus are written as escapes, not literally:
# this file is embedded in the notebook, and Kaggle's log viewer mangles any
# byte outside ASCII -- which turns a mangled traceback into a wasted hour.
# Python's re expands \uXXXX inside a pattern, so the match is unchanged.
_SCI = re.compile(r"(\d+(?:\.\d+)?)\s*[x\u00d7]\s*10\s*\^?\s*\(?\s*(-|\u2212)?\s*(\d+)")
_POW = re.compile(r"\^\s*-?\d+")
# A unit's exponent must go before any number is read. Without this "m/s^2"
# donates a stray 2.0, and 2.0 is the correct answer to two of the held-out
# problems -- so a wrong answer that merely wrote m/s^2 would score correct.
#
# No whitespace is allowed between the letter and the digits. Permitting it made
# this eat every number that merely followed a word: "is 10.67" matched from the
# "s", leaving ".67", and "about 2200000.0" became ".0". An exponent is always
# written tight against its unit.
#
# 'e' and 'E' are excluded from the lookbehind because _SCI rewrites "5.3 x
# 10^-11" as the float repr "5.3e-11"; treating that exponent as a unit would
# leave 5.3 behind, the same magnitude error this rule exists to prevent.
_UNIT_EXP = re.compile(r"(?<=[a-df-zA-DF-Z])\^?-?\d+")
# The exponent part is required: _SCI rewrites "5.3 x 10^-11" as the float repr
# "5.3e-11", and without it that reads as the two separate numbers 5.3 and -11.
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(text):
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    t = _SCI.sub(lambda m: repr(float(m.group(1)) * 10 ** (
        -int(m.group(3)) if m.group(2) else int(m.group(3)))), t)
    t = _POW.sub(" ", t)
    t = _UNIT_EXP.sub(" ", t)
    out = []
    for m in _NUM.finditer(t):
        try:
            out.append(float(m.group(0).replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def answer_span(text):
    """What the model presents as its answer, not its scratch work.

    Everything before `</think>` is exploration full of intermediate numbers, so
    scoring it would reward a model for merely mentioning the right value
    somewhere. When the block never closes -- which the untuned base does on
    every problem -- the whole text is the only thing available.
    """
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def score(text, parts):
    """A problem counts correct only when EVERY part is present.

    A multi-part answer is a multi-part question; giving credit for one of three
    would flatter every arm equally and hide exactly the differences worth
    seeing.
    """
    nums = numbers_in(answer_span(text))
    if not nums:
        nums = numbers_in(text)
    hits = []
    for p in parts:
        want, tol = p["value"], p.get("tol", 0.03)
        margin = max(abs(want) * tol, 1e-9)
        hits.append(any(abs(n - want) <= margin for n in nums))
    return all(hits), hits


def closes_think(text):
    return "</think>" in text


def reasoning_tokens(text, tok):
    """Tokens spent before the answer.

    The chat template ends the PROMPT with an open `<think>`, so generation
    begins already inside the block and the opening tag is not part of the
    output. Measuring to `</think>` is therefore the honest span; when there is
    no closing tag the model never left the block, and the whole output counts.
    """
    span = text.split("</think>")[0] if "</think>" in text else text
    span = span.replace("<think>", "")
    return len(tok(span, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------- training

def make_args(**kw):
    """TrainingArguments differ between transformers 4 and 5 on Kaggle.

    Unsupported keys are dropped rather than crashing the run, and warmup_ratio
    is converted to warmup_steps when only the latter exists.
    """
    import inspect
    ok = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "warmup_ratio" in kw and "warmup_ratio" not in ok:
        kw["warmup_steps"] = max(1, int(kw.pop("warmup_ratio") * kw.get("_total", 100)))
    kw.pop("_total", None)
    if "eval_strategy" not in ok and "evaluation_strategy" in ok:
        kw["evaluation_strategy"] = kw.pop("eval_strategy", "no")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in ok})


def build_example(row, tok, double_think):
    """One training example.

    `double_think` reproduces the ORIGINAL bug on purpose so the fixed and
    unfixed recipes can be compared on equal terms. The chat template already
    ends the prompt with `<think>\\n`; the original target opened the tag a
    second time, so every earlier fine-tune learned to emit a redundant
    `<think>` with its real reasoning nested inside a block that never closes.
    """
    prefix = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                     tokenize=False, add_generation_prompt=True)
    head = "<think>\n" if double_think else ""
    resp = f"{head}{row['think']}\n</think>\n\n{row['solution']}"
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    r = tok(resp, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": (p + r)[:MAX_LEN], "labels": ([-100] * len(p) + r)[:MAX_LEN]}


def train_adapter(a, tok, out_dir):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    rows = [json.loads(l) for l in
            Path(a.train_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    train_rows = [r for r in rows if r.get("split") == "train"]
    # The ten validation problems are the ones the original run early-stopped
    # on; reusing them keeps this comparable rather than inventing a new split.
    bundle = json.loads(Path(a.bundle).read_text(encoding="utf-8"))
    val_ids = {p["id"] for p in bundle["problems"] if p["source"] == "heldout_val"}
    val_rows = [r for r in rows if r["id"] in val_ids]
    print(f"[{a.arm}] train {len(train_rows)} | val {len(val_rows)} | "
          f"double_think={a.double_think}", flush=True)

    ds_tr = [build_example(r, tok, a.double_think) for r in train_rows]
    ds_va = [build_example(r, tok, a.double_think) for r in val_rows]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, lab, msk = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            lab.append(b["labels"] + [-100] * pad)
            msk.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(msk)}

    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map={"": 0}, **{dtype_kw: torch.float16})
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False

    ckpt = out_dir / f"_ckpt_{a.arm}"
    steps = max(1, math.ceil(len(ds_tr) / 4))
    args = make_args(
        output_dir=str(ckpt), num_train_epochs=a.epochs, learning_rate=2e-4,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        per_device_eval_batch_size=1, logging_steps=5, fp16=True,
        lr_scheduler_type="cosine", warmup_ratio=0.05, _total=steps * a.epochs,
        eval_strategy="steps", eval_steps=max(1, steps // 2),
        save_strategy="steps", save_steps=max(1, steps // 2), save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to=[], seed=0)
    tr = Trainer(model=model, args=args, train_dataset=ds_tr,
                 eval_dataset=ds_va, data_collator=collate)
    tr.train()
    adir = out_dir / a.arm
    model.save_pretrained(str(adir))
    hist = [h for h in tr.state.log_history if "eval_loss" in h]
    print(f"[{a.arm}] best eval_loss "
          f"{min((h['eval_loss'] for h in hist), default=None)}", flush=True)
    model.config.use_cache = True
    return model, str(adir), hist


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--double-think", action="store_true")
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=900)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"[{a.arm}]"
    print(f"{tag} {torch.cuda.get_device_name(0)} | transformers "
          f"{transformers.__version__}", flush=True)
    assert torch.cuda.get_device_capability(0)[0] >= 7

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    curve, adapter_dir = [], a.adapter
    if a.train:
        model, adapter_dir, curve = train_adapter(a, tok, out)
    else:
        dtype_kw = ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
                    else "torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True),
            device_map={"": 0}, **{dtype_kw: torch.float16})
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    # Left padding: a decoder-only model padded on the right puts the pad run
    # between the prompt and the first generated token, and every sequence in
    # the batch then decodes from the wrong position.
    tok.padding_side = "left"

    bundle = json.loads(Path(a.bundle).read_text(encoding="utf-8"))
    probs = bundle["problems"]
    print(f"{tag} scoring {len(probs)} problems", flush=True)

    prompts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                       add_generation_prompt=True, tokenize=False)
               for p in probs]

    rows, t0 = [], time.time()
    for i in range(0, len(probs), a.batch):
        enc = tok(prompts[i:i + a.batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        for j, text in enumerate(outs):
            p = probs[i + j]
            ok, hits = score(text, p["parts"])
            rows.append({"id": p["id"], "source": p["source"], "correct": bool(ok),
                         "part_hits": hits, "closes_think": closes_think(text),
                         "reasoning_tokens": reasoning_tokens(text, tok),
                         "opens_extra_think": text.lstrip().startswith("<think>"),
                         "chars": len(text),
                         "output": text if i + j < 12 else None})
        done = min(i + a.batch, len(probs))
        el = time.time() - t0
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"{tag} {done:4d}/{len(probs)}  acc {acc:.3f}  "
              f"{el / done:.1f}s/problem  eta {(len(probs) - done) * el / done / 60:.0f}m",
              flush=True)

    by = {}
    for r in rows:
        d = by.setdefault(r["source"], {"n": 0, "correct": 0})
        d["n"] += 1; d["correct"] += r["correct"]
    for s, d in by.items():
        d["accuracy"] = round(d["correct"] / d["n"], 4)

    med = sorted(r["reasoning_tokens"] for r in rows)
    res = {"arm": a.arm, "adapter_dir": adapter_dir, "trained": a.train,
           "double_think": a.double_think if a.train else None,
           "n": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
           "by_source": by,
           "closes_think": round(sum(r["closes_think"] for r in rows) / len(rows), 4),
           "extra_think_tag": round(sum(r["opens_extra_think"] for r in rows) / len(rows), 4),
           "median_reasoning_tokens": med[len(med) // 2],
           "minutes": round((time.time() - t0) / 60, 1),
           "curve": [{k: v for k, v in h.items() if k in ("epoch", "loss", "eval_loss")}
                     for h in curve],
           "per_problem": rows}
    (out / f"result_{a.arm}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{tag} DONE overall {res['accuracy']} | " +
          " | ".join(f"{s} {d['accuracy']}" for s, d in sorted(by.items())) +
          f" | extra tag {res['extra_think_tag']}", flush=True)


if __name__ == "__main__":
    main()
