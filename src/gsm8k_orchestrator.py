# %% [markdown]
# # LOOM on GSM8K -- A_v1_control against the untuned base
#
# GSM8K is not what LOOM was trained for. It is grade-school arithmetic; LOOM
# was trained on physics traces. The question this run answers is **not** "is
# LOOM good at maths" but **"did the fine-tune damage the general ability that
# was already there"**. A score near the base's is a pass. A score well below it
# means the adapter cost something.
#
# Both arms get the identical prompt, greedy decoding and token budget. The only
# difference is whether the adapter is attached.
#
# Set **Accelerator: GPU T4 x2** and **Internet: On**.

# %%
# !pip -q install -U "transformers>=4.44" accelerate peft "bitsandbytes>=0.46.1" datasets

# %%
from pathlib import Path
import json, os, subprocess, sys, time

OUT_DIR = Path("/kaggle/working/gsm8k")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORKER = Path("/kaggle/working/gsm8k_worker.py")

# The full official GSM8K test split. A subset would still measure damage, but
# only the complete 1319 produces a number that can be set beside a published
# GSM8K score without an asterisk.
N_PROBLEMS = 1319
# Batch 32 rather than 16: generation is decode-step bound, and the step cost
# grows far slower than the batch, so this buys back most of the time that going
# from 500 to 1319 problems costs. KV cache at this batch is about 1.5 GB
# against the T4's 15, so there is plenty of headroom.
BATCH = 32
MAX_NEW = 640


def _find(pattern, root="/kaggle/input"):
    hits = sorted(Path(root).glob(pattern))
    return hits[0] if hits else None


# The v1 adapter dataset also contains an adapter_config.json, so the search is
# anchored on this arm's own dataset slug rather than on the file name.
_cfg = _find("**/loom-a-v1-control-adapter/**/adapter_config.json") or \
       _find("**/adapter_config.json")
ADAPTER = _cfg.parent if _cfg else None
print("adapter ->", ADAPTER)
assert ADAPTER, "attach the loom-a-v1-control-adapter dataset"
print(json.loads((ADAPTER / "adapter_config.json").read_text())["base_model_name_or_path"])

import torch
NGPU = torch.cuda.device_count()
print(f"GPUs: {NGPU} -> " + ", ".join(torch.cuda.get_device_name(i) for i in range(NGPU)))
assert NGPU >= 1

# %%
WORKER_SRC = r'''__WORKER_SOURCE__'''
WORKER.write_text(WORKER_SRC, encoding="utf-8")
print(f"wrote {WORKER} ({len(WORKER_SRC):,} bytes)")

# %%
ARMS = [("BASE", None), ("A_v1_control", str(ADAPTER))]


def launch(arm, adapter, gpu):
    cmd = [sys.executable, str(WORKER), "--arm", arm, "--out", str(OUT_DIR),
           "--n", str(N_PROBLEMS), "--batch", str(BATCH), "--max-new", str(MAX_NEW)]
    if adapter:
        cmd += ["--adapter", adapter]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    log = (OUT_DIR / f"log_{arm}.txt").open("w", encoding="utf-8")
    print(f"  -> {arm} starting on GPU {gpu}", flush=True)
    return {"arm": arm, "log": log, "pos": 0,
            "proc": subprocess.Popen(cmd, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, text=True)}


def drain(job):
    p = OUT_DIR / f"log_{job['arm']}.txt"
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8", errors="replace")
    new, job["pos"] = txt[job["pos"]:], len(txt)
    for line in new.splitlines():
        if line.strip():
            print(line, flush=True)


queue = list(ARMS)
running, t0 = [], time.time()
while queue or running:
    while queue and len(running) < NGPU:
        arm, ad = queue.pop(0)
        running.append(launch(arm, ad, len(running) % NGPU))
    time.sleep(30)
    for job in list(running):
        drain(job)
        if job["proc"].poll() is not None:
            job["log"].close(); drain(job)
            print(f"== {job['arm']} exited rc={job['proc'].returncode} "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            running.remove(job)

print(f"\nboth arms finished in {(time.time() - t0) / 60:.0f} min")

# %% [markdown]
# ## Compare
#
# The number that matters is the **gap**, not either score on its own.

# %%
res = {}
for arm, _ in ARMS:
    f = OUT_DIR / f"gsm8k_{arm}.json"
    if f.exists():
        res[arm] = json.loads(f.read_text(encoding="utf-8"))
    else:
        print(f"!! {arm} produced no result -- see log_{arm}.txt")

hdr = f"{'arm':16s} {'accuracy':>9s} {'correct':>9s} {'closed':>7s} {'min':>6s}"
print(hdr); print("-" * len(hdr))
for arm, r in res.items():
    print(f"{arm:16s} {r['accuracy']:9.3f} {r['correct']:>4d}/{r['n']:<4d} "
          f"{r['closed_think_rate']:7.2f} {r['minutes']:6.1f}")

if len(res) == 2:
    b, a = res["BASE"], res["A_v1_control"]
    gap = a["accuracy"] - b["accuracy"]
    # Two independent proportions on the same n; the standard error of the
    # difference is what decides whether a gap means anything at all.
    import math
    se = math.sqrt(sum(r["accuracy"] * (1 - r["accuracy"]) / r["n"] for r in (b, a)))
    print(f"\ngap {gap * 100:+.1f} points, standard error {se * 100:.1f} points")
    if abs(gap) <= 2 * se:
        print("-> inside noise: no measurable damage to general ability")
    elif gap < 0:
        print("-> the fine-tune costs general ability by more than noise")
    else:
        print("-> the fine-tune helps here by more than noise")

Path("/kaggle/working/gsm8k_report.json").write_text(json.dumps({
    "created": time.strftime("%Y-%m-%d %H:%M"),
    "n": N_PROBLEMS, "max_new_tokens": MAX_NEW,
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_problem"}
                for k, v in res.items()},
    "per_problem": {k: v["per_problem"] for k, v in res.items()},
}, indent=2), encoding="utf-8")
print("\nwrote gsm8k_report.json")
