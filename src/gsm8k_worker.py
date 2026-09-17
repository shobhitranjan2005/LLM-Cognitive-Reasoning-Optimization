"""Score one arm on GSM8K, in its own process on its own GPU.

Same shape as loom_worker.py, and for the same reason: a process exit is the
only way to be sure every byte of VRAM comes back, and one worker per GPU lets
the base and the fine-tune run at the same time instead of one after the other.

    python gsm8k_worker.py --arm BASE --out /kaggle/working/gsm8k
    python gsm8k_worker.py --arm A_v1_control --adapter /path/to/adapter ...

Both arms get the identical prompt, identical decoding (greedy) and identical
token budget. The only difference is whether the adapter is attached.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# GSM8K gold answers close with "#### <number>".
GOLD = re.compile(r"####\s*([-\d,\.]+)")
# A number, with thousands separators and a trailing full stop allowed.
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gold_answer(text):
    m = GOLD.search(text)
    return float(m.group(1).replace(",", "").rstrip(".")) if m else None


def predicted_answer(text):
    """The model's final number.

    Everything before `</think>` is scratch work full of intermediate numbers,
    so the answer is looked for after it. R1-distill sometimes runs out of
    budget before closing the block; in that case the whole text is used, which
    is the honest fallback -- taking the last number of a truncated trace is
    what a reader would do.
    """
    tail = text.split("</think>")[-1]
    if not NUM.search(tail):
        tail = text
    hits = NUM.findall(tail)
    if not hits:
        return None
    try:
        return float(hits[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=640)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"[{a.arm}]"
    dev = torch.cuda.get_device_name(0)
    print(f"{tag} {dev} | adapter={a.adapter or 'none'}", flush=True)
    assert torch.cuda.get_device_capability(0)[0] >= 7, "needs a T4 or better"

    tok = AutoTokenizer.from_pretrained(BASE_ID)
    # Decoder-only batched generation must pad on the LEFT, otherwise the pad
    # run sits between the prompt and the first generated token and every
    # sequence in the batch decodes from the wrong position.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True),
        device_map={"": 0}, torch_dtype=torch.float16)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    print(f"{tag} model ready", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    total = len(ds)
    n = min(a.n, total)
    # A fixed shuffle, so both arms see the same problems in the same order and
    # a subset is not just the front of the file.
    ds = ds.shuffle(seed=0).select(range(n))
    print(f"{tag} {n} of {total} GSM8K test problems", flush=True)

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": r["question"]}],
        add_generation_prompt=True, tokenize=False) for r in ds]
    golds = [gold_answer(r["answer"]) for r in ds]

    rows, correct, truncated, t0 = [], 0, 0, time.time()
    for i in range(0, n, a.batch):
        batch = prompts[i:i + a.batch]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=a.max_new,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        for j, text in enumerate(outs):
            k = i + j
            pred, gold = predicted_answer(text), golds[k]
            ok = pred is not None and gold is not None and abs(pred - gold) < 1e-4
            correct += ok
            closed = "</think>" in text
            truncated += not closed
            rows.append({"i": k, "gold": gold, "pred": pred, "correct": bool(ok),
                         "closed_think": closed, "chars": len(text),
                         "output": text if k < 20 else None})
        done = min(i + a.batch, n)
        el = time.time() - t0
        print(f"{tag} {done:4d}/{n}  acc {correct / done:.3f}  "
              f"{el / done:.1f}s/problem  eta {(n - done) * el / done / 60:.0f}m",
              flush=True)

    res = {"arm": a.arm, "adapter": a.adapter, "n": n,
           "accuracy": round(correct / n, 4), "correct": correct,
           "closed_think_rate": round(1 - truncated / n, 4),
           "max_new_tokens": a.max_new, "minutes": round((time.time() - t0) / 60, 1),
           "per_problem": rows}
    (out / f"gsm8k_{a.arm}.json").write_text(json.dumps(res, indent=2),
                                             encoding="utf-8")
    print(f"{tag} DONE accuracy {res['accuracy']} "
          f"({correct}/{n}) in {res['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
