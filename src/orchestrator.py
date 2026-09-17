# %% [markdown]
# # LOOM v2 -- sweep across both T4s
#
# Each arm runs in its **own process on its own GPU**. Two reasons:
#
# 1. **VRAM.** In-process cleanup between arms leaked until arm 3 OOM'd
#    mid-training. A process exit returns every byte to the driver, so the leak
#    cannot accumulate.
# 2. **Speed.** Kaggle's "T4 x2" is two cards; the previous runs used one and
#    left the other idle. Two workers halves the wall clock.
#
# Finished arms are read back from the seeded results, so nothing already
# measured is recomputed. Set **Accelerator: GPU T4 x2** and **Internet: On**.

# %%
# Kaggle's image ships an older bitsandbytes than 4-bit QLoRA needs. The workers
# are separate processes but share this interpreter's site-packages, so
# installing here covers them too.
# !pip -q install -U "transformers>=4.44" accelerate peft "bitsandbytes>=0.46.1" datasets

# %%
from pathlib import Path
import json, os, subprocess, sys, time

OUT_DIR = Path("/kaggle/working/loom_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORKER = Path("/kaggle/working/loom_worker.py")


def _find(pattern, root="/kaggle/input"):
    hits = sorted(Path(root).glob(pattern))
    return hits[0] if hits else None


TRAIN_JSONL = _find("**/train.jsonl")
_v1cfg = _find("**/adapter_config.json")
V1_ADAPTER = _v1cfg.parent if _v1cfg else None
SEED_RESULTS = _find("**/sweep_results.json")

ALL_ARMS = ["BASE_untuned", "V1_original", "A_v1_control", "B_half", "C_attn_only",
            "D_lowrank", "E_gentle", "F_neftune", "G_rslora", "H_regularised"]

print("train.jsonl ->", TRAIN_JSONL)
print("v1 adapter  ->", V1_ADAPTER)
print("seed results->", SEED_RESULTS)
assert TRAIN_JSONL, "train.jsonl not found under /kaggle/input"

import torch
NGPU = torch.cuda.device_count()
print(f"GPUs visible: {NGPU} -> " +
      ", ".join(torch.cuda.get_device_name(i) for i in range(NGPU)))
assert NGPU >= 1

# %%
# The worker source is embedded so this notebook is the only thing to upload.
WORKER_SRC = r'''__WORKER_SOURCE__'''
WORKER.write_text(WORKER_SRC, encoding="utf-8")
print(f"wrote {WORKER} ({len(WORKER_SRC):,} bytes)")

# %%
RESULTS = {}
if SEED_RESULTS:
    RESULTS = json.loads(SEED_RESULTS.read_text(encoding="utf-8"))
    RESULTS.pop("_gsm8k", None)
    print(f"seeded {len(RESULTS)} arms: {sorted(RESULTS)}")
for f in OUT_DIR.glob("result_*.json"):        # anything this session already did
    r = json.loads(f.read_text(encoding="utf-8"))
    RESULTS[r["label"]] = r

PENDING = [a for a in ALL_ARMS if a not in RESULTS]
if V1_ADAPTER is None and "V1_original" in PENDING:
    PENDING.remove("V1_original")
    print("no v1 adapter attached -- skipping that comparison")
print(f"\npending ({len(PENDING)}): {PENDING}")

# %%
def launch(arm, gpu):
    cmd = [sys.executable, str(WORKER), "--arm", arm,
           "--train-jsonl", str(TRAIN_JSONL), "--out", str(OUT_DIR)]
    if V1_ADAPTER:
        cmd += ["--v1-adapter", str(V1_ADAPTER)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log = (OUT_DIR / f"log_{arm}.txt").open("w", encoding="utf-8")
    print(f"  -> {arm} starting on GPU {gpu}", flush=True)
    return {"arm": arm, "gpu": gpu, "log": log, "pos": 0,
            "proc": subprocess.Popen(cmd, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, text=True)}


def drain(job):
    """Echo whatever the worker has written since last time, tagged by arm."""
    p = OUT_DIR / f"log_{job['arm']}.txt"
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8", errors="replace")
    new, job["pos"] = txt[job["pos"]:], len(txt)
    for line in new.splitlines():
        if line.strip():
            print(f"[{job['arm']}] {line}", flush=True)


queue, running, t0 = list(PENDING), [], time.time()
while queue or running:
    while queue and len(running) < NGPU:
        busy = {j["gpu"] for j in running}
        gpu = next(g for g in range(NGPU) if g not in busy)
        running.append(launch(queue.pop(0), gpu))

    time.sleep(20)
    for job in list(running):
        drain(job)
        if job["proc"].poll() is not None:
            job["log"].close(); drain(job)
            rc = job["proc"].returncode
            f = OUT_DIR / f"result_{job['arm']}.json"
            if rc == 0 and f.exists():
                RESULTS[job["arm"]] = json.loads(f.read_text(encoding="utf-8"))
                print(f"== {job['arm']} DONE ({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
            else:
                print(f"== {job['arm']} FAILED (rc={rc}) -- see log_{job['arm']}.txt", flush=True)
            running.remove(job)
            (OUT_DIR / "sweep_results.json").write_text(
                json.dumps(RESULTS, indent=2), encoding="utf-8")

print(f"\nsweep finished in {(time.time()-t0)/60:.0f} min; {len(RESULTS)} arms have results")

# %% [markdown]
# ## Pick the winner
#
# 1. **Damage gate** -- closes `</think>` on >= 90% of problems, general ability
#    within 10 points of base, perplexity <= 1.25x base
# 2. Highest **accuracy** on the 10 pristine problems
# 3. Ties: lower **regurgitation**, then fewer reasoning tokens

# %%
BASE = RESULTS.get("BASE_untuned", {})
base_gen, base_ppl = BASE.get("general_score", 0.0), BASE.get("perplexity", float("inf"))

cands = []
for name, r in RESULTS.items():
    if name in ("BASE_untuned", "V1_original"):
        continue
    fail = []
    if r["closes_think"] < 0.9:
        fail.append("format")
    if r["general_score"] < base_gen - 0.10:
        fail.append("general")
    if r["perplexity"] > base_ppl * 1.25:
        fail.append("perplexity")
    cands.append({**r, "name": name, "eligible": not fail, "failed": fail})

hdr = (f"{'arm':16s} {'acc':>5s} {'close':>6s} {'reason':>7s} {'ngram':>6s} "
       f"{'gen':>5s} {'ppl':>7s} {'evalL':>7s}  status")
print(hdr); print("-" * len(hdr))
for k in ("BASE_untuned", "V1_original"):
    if k in RESULTS:
        r = RESULTS[k]
        print(f"{k:16s} {r['accuracy']:5.2f} {r['closes_think']:6.2f} "
              f"{str(r['median_reasoning_tokens']):>7s} {r['max_shared_ngram']:6d} "
              f"{r['general_score']:5.2f} {r['perplexity']:7.2f} {'-':>7s}  reference")
for r in sorted(cands, key=lambda x: x["name"]):
    print(f"{r['name']:16s} {r['accuracy']:5.2f} {r['closes_think']:6.2f} "
          f"{str(r['median_reasoning_tokens']):>7s} {r['max_shared_ngram']:6d} "
          f"{r['general_score']:5.2f} {r['perplexity']:7.2f} "
          f"{(r.get('best_eval_loss') or 0):7.3f}  "
          f"{'eligible' if r['eligible'] else 'DAMAGED: ' + ','.join(r['failed'])}")

ranked = sorted([c for c in cands if c["eligible"]],
                key=lambda r: (-r["accuracy"], r["max_shared_ngram"],
                               r["median_reasoning_tokens"] or 10**6))
assert ranked, "no candidate passed the damage gate"
WINNER = ranked[0]
print(f"\nWINNER: {WINNER['name']} | accuracy {WINNER['accuracy']} | "
      f"regurgitation {WINNER['max_shared_ngram']} words | "
      f"reasoning {WINNER['median_reasoning_tokens']} tokens")

# %%
import shutil

BEST = OUT_DIR / "BEST"
if BEST.exists():
    shutil.rmtree(BEST)
shutil.copytree(WINNER["adapter_dir"], BEST)
(BEST / "PROVENANCE.md").write_text(f"""# LOOM adapter v2 -- {WINNER['name']}

Trained {time.strftime('%Y-%m-%d %H:%M')} on Kaggle T4. A NEW adapter; it does not
replace v1, which stays archived at models/v1_original_2026-08-07/.

## Recipe
{json.dumps(WINNER.get('config', {}), indent=2)}

## Measured on 10 pristine held-out problems
(never trained on, never used to pick a checkpoint)

- accuracy                {WINNER['accuracy']}   (base {BASE.get('accuracy')}, v1 {RESULTS.get('V1_original',{}).get('accuracy')})
- closes think block      {WINNER['closes_think']}
- median reasoning tokens {WINNER['median_reasoning_tokens']}
- longest word run shared with any training trace: {WINNER['max_shared_ngram']}
- general-ability probes  {WINNER['general_score']} (base {base_gen})
- perplexity              {WINNER['perplexity']} (base {base_ppl})
- best eval loss {WINNER.get('best_eval_loss')} / final train loss {WINNER.get('final_train_loss')}

Full detail, every generated answer and every loss curve: v2_report.json
""", encoding="utf-8")

Path("/kaggle/working/v2_report.json").write_text(json.dumps({
    "created": time.strftime("%Y-%m-%d %H:%M"),
    "winner": WINNER["name"],
    "ranking": [r["name"] for r in ranked],
    "results": RESULTS,
}, indent=2), encoding="utf-8")

shutil.make_archive("/kaggle/working/loom_v2_best", "zip", str(BEST))
print("wrote loom_v2_best.zip and v2_report.json")
