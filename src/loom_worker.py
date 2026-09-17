"""Train and grade ONE sweep arm, in its own process, on one GPU.

Run per-arm in a subprocess for two reasons:

1. **VRAM.** In-process cleanup between arms leaked until the third arm OOM'd
   mid-training. A process exit hands every byte back to the driver, so a leak
   cannot accumulate across arms no matter what PEFT/Trainer hold onto.
2. **Both T4s.** Kaggle's "T4 x2" is two cards. One worker per card halves the
   wall clock, and CUDA_VISIBLE_DEVICES keeps them from seeing each other.

    python loom_worker.py --arm B_half --train-jsonl ... --out DIR [--v1-adapter DIR]

Writes DIR/result_<arm>.json. Arms "BASE_untuned" and "V1_original" grade only.
"""
import argparse
import gc
import inspect
import json
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_LEN = 768
SEED = 20260819

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
                         modules="all", rslora=True, note="rank-stabilised LoRA scaling"),
    "H_regularised": dict(r=8, alpha=16, dropout=0.10, lr=1e-4, epochs=6,
                          modules="all", wd=0.05, label_smoothing=0.05,
                          note="weight decay + label smoothing"),
}

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

# ------------------------------------------------------------------ grading

_SCI = re.compile(r"(\d+(?:\.\d+)?)\s*[x*]\s*10\s*\^?\s*\(?(-?\d+)\)?")
_POW = re.compile(r"(\d+(?:\.\d+)?)\s*\^\s*\(?(-?\d+)\)?")
_UNIT_EXP = re.compile(r"(?<=[A-Za-z)])\s*\^\s*\(?-?\d+\)?")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(text):
    """Numbers in `text`; '1,200' and '5 x 10^5' understood, unit exponents ignored.

    Stripping 'm/s^2' matters: it otherwise donates a stray 2.0, and 2.0 is the
    right answer to two held-out problems, so wrong answers would score correct.
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
    i = text.rfind("</think>")
    return text[i + len("</think>"):] if i != -1 else text


def is_correct(text, parts):
    nums = numbers_in(answer_span(text))
    if not nums:
        return False
    for _n, want, _u, tol in parts:
        if not any(math.isclose(v, want, rel_tol=tol, abs_tol=abs(want) * tol + 1e-9)
                   for v in nums):
            return False
    return True


def reasoning_tokens(text, tok):
    """Tokens of reasoning before the answer.

    The chat template ends the PROMPT with an open `<think>`, so generation starts
    already inside the block and the opening tag never appears in the output. The
    honest measure is therefore everything up to `</think>` -- requiring both tags
    scores a properly-reasoning model as zero.
    """
    i = text.find("</think>")
    body = text[:i] if i != -1 else ""
    return len(tok(body, add_special_tokens=False)["input_ids"]) if i != -1 else None


def closes_think(text):
    return "</think>" in text


_NGRAM_SIZES = [5, 8, 12, 20, 30, 45, 60]


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def build_corpus_index(rows):
    idx = {n: set() for n in _NGRAM_SIZES}
    for r in rows:
        w = _words(f"{r['think']} {r['solution']}")
        for n in _NGRAM_SIZES:
            for i in range(len(w) - n + 1):
                idx[n].add(tuple(w[i:i + n]))
    return idx


def max_shared_ngram(text, idx):
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


def split_heldout(held):
    by_ch = {}
    for r in held:
        by_ch.setdefault(r["chapter"], []).append(r)
    val, pristine = [], []
    for i, ch in enumerate(sorted(by_ch)):
        ids = sorted(by_ch[ch], key=lambda r: r["id"])
        first, second = [ids[0], ids[2], ids[4]], [ids[1], ids[3]]
        if i % 2 == 0:
            val += first; pristine += second
        else:
            val += second; pristine += first
    return (sorted(val, key=lambda r: r["id"]),
            sorted(pristine, key=lambda r: r["id"]))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1-adapter", default=None)
    a = ap.parse_args()

    import torch
    import transformers
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from transformers.trainer_callback import EarlyStoppingCallback
    from peft import (LoraConfig, PeftModel, get_peft_model,
                      prepare_model_for_kbit_training)

    torch.manual_seed(SEED)
    cc = torch.cuda.get_device_capability(0)
    assert cc[0] >= 7, f"{torch.cuda.get_device_name(0)} is sm_{cc[0]}{cc[1]}; needs sm_70+"
    NATIVE_BF16 = cc[0] >= 8
    DTYPE = torch.bfloat16 if NATIVE_BF16 else torch.float16
    print(f"[{a.arm}] {torch.cuda.get_device_name(0)} | "
          f"{'bf16' if NATIVE_BF16 else 'fp16'} | transformers {transformers.__version__}",
          flush=True)

    rows = [json.loads(l) for l in Path(a.train_jsonl).open(encoding="utf-8")]
    train_rows = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "test"]
    val_rows, pristine_rows = split_heldout(held)
    ngram_idx = build_corpus_index(train_rows)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

    def build(row):
        prefix = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                         tokenize=False, add_generation_prompt=True)
        resp = f"<think>\n{row['think']}\n</think>\n\n{row['solution']}"
        p = tok(prefix, add_special_tokens=False)["input_ids"]
        r = tok(resp, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        return {"input_ids": (p + r)[:MAX_LEN], "labels": ([-100] * len(p) + r)[:MAX_LEN]}

    train_ds = [build(r) for r in train_rows]
    val_ds = [build(r) for r in val_rows]

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

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE)

    def fresh_base():
        m = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map={"": 0}, **{dtype_kw: DTYPE})
        m.config.use_cache = False
        return m

    @torch.no_grad()
    def generate(model, prompt, max_new=700):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(0)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def perplexity(model):
        ids = tok(PPL_TEXT, return_tensors="pt").to(0)
        return float(torch.exp(model(**ids, labels=ids["input_ids"]).loss))

    def grade(model, label):
        model.eval()
        model.config.use_cache = True
        per, ok, closed, toks, ngr = [], 0, 0, [], []
        for r in pristine_rows:
            t0 = time.time()
            gen = generate(model, r["prompt"])
            c = is_correct(gen, ANSWER_KEY[r["id"]])
            rt = reasoning_tokens(gen, tok)
            cl = closes_think(gen)
            ng = max_shared_ngram(gen, ngram_idx)
            ok += c; closed += cl
            if rt:
                toks.append(rt)
            ngr.append(ng)
            per.append({"id": r["id"], "correct": bool(c), "closes_think": bool(cl),
                        "reasoning_tokens": rt, "max_shared_ngram": ng,
                        "seconds": round(time.time() - t0, 1), "output": gen})
            print(f"  {r['id']:9s} {'OK ' if c else '.  '} close={int(cl)} "
                  f"reason={rt} ngram={ng} ({time.time() - t0:.0f}s)", flush=True)

        pok, pdet = 0, []
        for q, exp in PROBES:
            g = generate(model, q, max_new=400)
            hit = any(e in g.lower() for e in exp)
            pok += hit
            pdet.append({"q": q, "hit": bool(hit),
                         "reasoning_tokens": reasoning_tokens(g, tok), "output": g[:600]})

        rep = {"label": label, "n": len(pristine_rows), "correct": ok,
               "accuracy": round(ok / len(pristine_rows), 3),
               "closes_think": round(closed / len(pristine_rows), 3),
               "median_reasoning_tokens": (sorted(toks)[len(toks) // 2] if toks else None),
               "max_shared_ngram": max(ngr) if ngr else 0,
               "general_score": round(pok / len(PROBES), 3),
               "perplexity": round(perplexity(model), 3),
               "per_problem": per, "probes": pdet}
        print(f"[{label}] acc {rep['accuracy']} | close {rep['closes_think']} | "
              f"reason~{rep['median_reasoning_tokens']} | ngram {rep['max_shared_ngram']} | "
              f"general {rep['general_score']} | ppl {rep['perplexity']}", flush=True)
        return rep

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    if a.arm == "BASE_untuned":
        rep = grade(fresh_base(), a.arm)
    elif a.arm == "V1_original":
        rep = grade(PeftModel.from_pretrained(fresh_base(), a.v1_adapter), a.arm)
    else:
        cfg = SWEEP[a.arm]
        print(f"[{a.arm}] {cfg.get('note','')}", flush=True)
        base = prepare_model_for_kbit_training(fresh_base(), use_gradient_checkpointing=True)
        base.gradient_checkpointing_enable(); base.enable_input_require_grads()

        attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_kw = dict(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                       bias="none", task_type="CAUSAL_LM",
                       target_modules=(attn if cfg["modules"] == "attn"
                                       else attn + ["gate_proj", "up_proj", "down_proj"]))
        if cfg.get("rslora"):
            lora_kw["use_rslora"] = True
        model = get_peft_model(base, LoraConfig(**lora_kw))
        model.print_trainable_parameters()

        spe = max(1, math.ceil(len(train_ds) / 4))
        every = max(1, spe // 2)
        early = cfg.get("early_stop", True)
        ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
        eval_kw = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

        kw = {"output_dir": str(out_dir / f"_ckpt_{a.arm}"),
              "num_train_epochs": cfg["epochs"], "per_device_train_batch_size": 1,
              "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 4,
              "learning_rate": cfg["lr"], "lr_scheduler_type": "cosine",
              "warmup_ratio": 0.05, "logging_steps": 5,
              eval_kw: "steps", "eval_steps": every,
              "save_strategy": "steps", "save_steps": every, "save_total_limit": 1,
              "load_best_model_at_end": early, "metric_for_best_model": "eval_loss",
              "greater_is_better": False, "weight_decay": cfg.get("wd", 0.0),
              "label_smoothing_factor": cfg.get("label_smoothing", 0.0),
              "fp16": not NATIVE_BF16, "bf16": NATIVE_BF16,
              "optim": "paged_adamw_8bit", "gradient_checkpointing": True,
              "report_to": "none", "seed": SEED}
        if cfg.get("neftune") and "neftune_noise_alpha" in ta_params:
            kw["neftune_noise_alpha"] = cfg["neftune"]
        # Kaggle's transformers moves fast (5.15 dropped warmup_ratio); adapt rather
        # than pin, and keep the warmup by converting it to steps.
        if "warmup_ratio" not in ta_params and "warmup_steps" in ta_params:
            kw["warmup_steps"] = max(1, int(spe * cfg["epochs"] * kw.pop("warmup_ratio")))
        dropped = sorted(k for k in kw if k not in ta_params)
        if dropped:
            print(f"  [TrainingArguments] dropped unsupported: {dropped}", flush=True)
        args = TrainingArguments(**{k: v for k, v in kw.items() if k in ta_params})

        trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                          eval_dataset=val_ds, data_collator=collate,
                          callbacks=[EarlyStoppingCallback(4)] if early else [])
        trainer.train()

        adapter_dir = out_dir / a.arm
        model.save_pretrained(str(adapter_dir))
        print(f"saved -> {adapter_dir}", flush=True)

        hist = trainer.state.log_history
        evals = [h["eval_loss"] for h in hist if "eval_loss" in h]
        trains = [h["loss"] for h in hist if "loss" in h]
        rep = grade(model, a.arm)
        rep.update({"config": dict(cfg), "adapter_dir": str(adapter_dir),
                    "best_eval_loss": min(evals) if evals else None,
                    "final_eval_loss": evals[-1] if evals else None,
                    "final_train_loss": trains[-1] if trains else None,
                    "eval_every_steps": every, "steps_per_epoch": spe,
                    "curve": [{k: v for k, v in h.items()
                               if k in ("epoch", "loss", "eval_loss")} for h in hist
                              if "loss" in h or "eval_loss" in h]})

    (out_dir / f"result_{a.arm}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[{a.arm}] wrote result_{a.arm}.json", flush=True)
    gc.collect()


if __name__ == "__main__":
    main()
