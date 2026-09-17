# %% [markdown]
# # LOOM physics evaluation -- and the `<think>` tag fix
#
# Four arms, scored on the same 150 problems:
#
# | arm | what it is |
# |---|---|
# | `BASE` | untouched DeepSeek-R1-Distill-Qwen-7B |
# | `V1_original` | the adapter currently in the prototype |
# | `A_v1_control` | the sweep winner, 8/10 -- trained with the duplicate `<think>` |
# | `A_fixed` | the same recipe with the tag bug fixed, trained here |
#
# The bundle is 10 pristine held-out physics problems (never trained on, never
# used to pick a checkpoint), 10 validation problems (held out of training but
# used for early stopping, so scores there are optimistic), and 130 curated
# HC Verma problems written by someone other than us.
#
# The 10 pristine problems are the identical set the earlier sweep scored, so
# `BASE 3/10`, `V1 6/10` and `A_v1_control 8/10` must reproduce. If they do not,
# something in this harness is wrong and nothing else here can be trusted.
#
# Set **Accelerator: GPU T4 x2** and **Internet: On**.

# %%
# !pip -q install -U "transformers>=4.44" accelerate peft "bitsandbytes>=0.46.1" datasets

# %%
from pathlib import Path
import json, os, subprocess, sys, time

OUT_DIR = Path("/kaggle/working/phys")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORKER = Path("/kaggle/working/physics_worker.py")

BATCH = 12
MAX_NEW = 900
EPOCHS = 8


def find(pattern, root="/kaggle/input"):
    hits = sorted(Path(root).glob(pattern))
    return hits[0] if hits else None


# Both adapter datasets contain a file called adapter_config.json, so each is
# located by its own dataset slug. Matching on the file name alone would silently
# score one adapter twice.
BUNDLE = find("**/eval_bundle.json")
TRAIN = find("**/train.jsonl")
_a = find("**/loom-a-v1-control-adapter/**/adapter_config.json")
_v1 = find("**/deep-reason/**/adapter_config.json")
A_ADAPTER = _a.parent if _a else None
V1_ADAPTER = _v1.parent if _v1 else None

print("bundle      ->", BUNDLE)
print("train.jsonl ->", TRAIN)
print("A adapter   ->", A_ADAPTER)
print("v1 adapter  ->", V1_ADAPTER)
assert BUNDLE, "attach loom-eval-bundle"
assert TRAIN, "attach loom-train"

b = json.loads(BUNDLE.read_text(encoding="utf-8"))
print("bundle:", b["count"], b["by_source"])

import torch
NGPU = torch.cuda.device_count()
print(f"GPUs: {NGPU} -> " + ", ".join(torch.cuda.get_device_name(i) for i in range(NGPU)))
assert NGPU >= 1

# %%
WORKER_SRC = r'''__WORKER_SOURCE__'''
WORKER.write_text(WORKER_SRC, encoding="utf-8")
print(f"wrote {WORKER} ({len(WORKER_SRC):,} bytes)")

# %%
# A_fixed is trained here and takes the longest, so it starts first. BASE is the
# slowest to score -- it never closes its think block and runs to the cap -- so
# it starts on the other card at the same time.
JOBS = [
    {"arm": "A_fixed", "train": True},
    {"arm": "BASE"},
    {"arm": "V1_original", "adapter": V1_ADAPTER},
    {"arm": "A_v1_control", "adapter": A_ADAPTER},
]
JOBS = [j for j in JOBS if j.get("train") or j.get("adapter") or j["arm"] == "BASE"]
print("queue:", [j["arm"] for j in JOBS])


def launch(job, gpu):
    cmd = [sys.executable, str(WORKER), "--arm", job["arm"],
           "--bundle", str(BUNDLE), "--out", str(OUT_DIR),
           "--batch", str(BATCH), "--max-new", str(MAX_NEW)]
    if job.get("train"):
        # no --double-think: this is the fix
        cmd += ["--train", "--train-jsonl", str(TRAIN), "--epochs", str(EPOCHS)]
    if job.get("adapter"):
        cmd += ["--adapter", str(job["adapter"])]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log = (OUT_DIR / f"log_{job['arm']}.txt").open("w", encoding="utf-8")
    print(f"  -> {job['arm']} starting on GPU {gpu}", flush=True)
    return {"arm": job["arm"], "log": log, "pos": 0,
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


queue, running, t0 = list(JOBS), [], time.time()
while queue or running:
    while queue and len(running) < NGPU:
        busy = {j["gpu"] for j in running}
        gpu = next(g for g in range(NGPU) if g not in busy)
        j = launch(queue.pop(0), gpu)
        j["gpu"] = gpu
        running.append(j)
    time.sleep(30)
    for job in list(running):
        drain(job)
        if job["proc"].poll() is not None:
            job["log"].close(); drain(job)
            print(f"== {job['arm']} exited rc={job['proc'].returncode} "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            running.remove(job)

print(f"\nall arms finished in {(time.time() - t0) / 60:.0f} min")

# %% [markdown]
# ## Results
#
# Wrapped so that a formatting mistake cannot destroy a run that has already
# done its work -- the last sweep lost its packaging step to a `KeyError` after
# 70 minutes of correct training.

# %%
try:
    res = {}
    for f in sorted(OUT_DIR.glob("result_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        res[r["arm"]] = r

    hdr = (f"{'arm':14s} {'pristine':>9s} {'val':>6s} {'hcverma':>8s} {'all':>6s} "
           f"{'close':>6s} {'2xtag':>6s} {'reason':>7s} {'min':>6s}")
    print(hdr); print("-" * len(hdr))
    for arm in ["BASE", "V1_original", "A_v1_control", "A_fixed"]:
        r = res.get(arm)
        if not r:
            continue
        s = r["by_source"]
        def acc(k):
            return f"{s[k]['accuracy']:.2f}" if k in s else "  -  "
        print(f"{arm:14s} {acc('heldout_pristine'):>9s} {acc('heldout_val'):>6s} "
              f"{acc('hcverma'):>8s} {r['accuracy']:6.3f} {r['closes_think']:6.2f} "
              f"{r['extra_think_tag']:6.2f} {r['median_reasoning_tokens']:7d} "
              f"{r['minutes']:6.1f}")

    # The earlier sweep scored these same ten problems. If the numbers moved,
    # this harness disagrees with the one that produced every banked result.
    print("\nreproduction check against the banked sweep (10 pristine):")
    for arm, want in [("BASE", 0.3), ("V1_original", 0.6), ("A_v1_control", 0.8)]:
        r = res.get(arm)
        if not r or "heldout_pristine" not in r["by_source"]:
            continue
        got = r["by_source"]["heldout_pristine"]["accuracy"]
        print(f"  {arm:14s} banked {want:.1f}  now {got:.1f}  "
              f"{'MATCH' if abs(got - want) < 1e-6 else 'DIFFERS -- investigate'}")

    import math
    if "hcverma" in res.get("A_fixed", {}).get("by_source", {}) and \
       "hcverma" in res.get("A_v1_control", {}).get("by_source", {}):
        x = res["A_fixed"]["by_source"]["hcverma"]
        y = res["A_v1_control"]["by_source"]["hcverma"]
        gap = x["accuracy"] - y["accuracy"]
        se = math.sqrt(sum(d["accuracy"] * (1 - d["accuracy"]) / d["n"] for d in (x, y)))
        print(f"\nA_fixed vs A_v1_control on HC Verma: {gap * 100:+.1f} points, "
              f"standard error {se * 100:.1f}")
        print("  -> " + ("inside noise" if abs(gap) <= 2 * se else "a real difference"))

    Path("/kaggle/working/physics_report.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_problem"}
                    for k, v in res.items()},
        "per_problem": {k: v["per_problem"] for k, v in res.items()},
    }, indent=2), encoding="utf-8")
    print("\nwrote physics_report.json")
except Exception as e:
    import traceback
    print("REPORTING FAILED, but every result_*.json is already on disk:")
    traceback.print_exc()
