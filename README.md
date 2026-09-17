# LOOM v2 — how to run the training on Kaggle

Everything is in **`notebooks/loom_v2_train.ipynb`**. Import it, click Run All, come back in
2–3.5 hours. `src/loom_v2_train.py` is the same code in script form (that is the file
to edit); `src/test_grading_logic.py` re-checks the grading offline with no GPU.

## Setup — 5 minutes

1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** →
   upload `notebooks/loom_v2_train.ipynb`
2. Right panel → **Session options**
   - **Accelerator: GPU T4** ← the notebook refuses to start without this
   - **Internet: On** (needed for the base model download)
3. Right panel → **Input → Add Input → Upload a Dataset** → upload `train.jsonl`
   (project root, 72 rows). Name it `loom-train`.
4. *(optional but worth it)* Add your existing `deep-reason-lora` dataset too, so
   the run benchmarks v2 against v1 on identical problems.
5. Check the two paths in the first cell match your dataset slugs:
   ```python
   TRAIN_JSONL = Path("/kaggle/input/loom-train/train.jsonl")
   V1_ADAPTER  = Path("/kaggle/input/deep-reason-lora/deep_reason_lora")
   ```
6. **Run All.**

## What it does, in order

| Stage | Time | What happens |
|---|---|---|
| Baselines | ~20 min | Grades the untouched base model, and v1 if mounted |
| Sweep | ~1.5–2 h | Trains 8 LoRA arms, each with per-epoch validation and early stopping |
| Grading | ~30 min | Every arm on 10 pristine problems + 12 general-ability probes |
| Package | 1 min | Zips the winner |

**It is resumable.** If the session dies, re-run the cells — finished arms are
read back from `sweep_results.json` and skipped.

**To go faster**, trim `RUN_ARMS` in the first cell.

## What to download when it finishes

From the **Output** panel (right side):

- **`loom_v2_best.zip`** — the winning adapter, plus `PROVENANCE.md` recording
  the recipe and every number behind the choice
- **`v2_report.json`** — the full run: loss curves for all 8 arms, every
  generated answer, all metrics

Send both back and I'll write up the comparison against v1 and wire the winner
into the demo — the prototype needs a GGUF conversion step before it can load it.

## How the winner is chosen

Selection runs in code, before anyone sees the numbers, so it can't be fudged:

1. **Damage gate** — format compliance ≥ 0.9, general ability within 10 points of
   base, perplexity ≤ 1.25× base. Anything that fails is marked `DAMAGED` and
   cannot win, no matter how good its physics looks.
2. Among survivors: **highest accuracy** on the 10 pristine problems.
3. Ties: **lower regurgitation** (shortest word run copied from training), then
   fewer think tokens.

## The splits

The 20 held-out `c` variants are cut 10/10, deterministically, stratified by chapter:

- **val** (early stopping, checkpoint selection) — `1D-01-c 1D-03-c 1D-05-c
  2D-02-c 2D-04-c CM-01-c CM-03-c CM-05-c FO-02-c FO-04-c`
- **pristine** (never trained on, never used to select anything — every reported
  number comes from here) — `1D-02-c 1D-04-c 2D-01-c 2D-03-c 2D-05-c CM-02-c
  CM-04-c FO-01-c FO-03-c FO-05-c`

`train.jsonl` is read only. Its `split` field is untouched; the 10/10 grouping
exists only in memory during the run.

## The sweep arms

| Arm | Attack on memorisation |
|---|---|
| `A_v1_control` | none — the v1 recipe with eval logging added and early stopping **off**. The control that shows the overfit curve |
| `B_half` | half the rank, half the learning rate |
| `C_attn_only` | attention only, no MLP capacity |
| `D_lowrank` | rank 4 — too small to store 52 traces |
| `E_gentle` | v1 capacity, quarter learning rate |
| `F_neftune` | NEFTune embedding noise, designed for small SFT sets |
| `G_rslora` | rank-stabilised LoRA scaling |
| `H_regularised` | weight decay + label smoothing |

## Safety

- The base model is **reloaded fresh for every arm**, so one bad arm cannot
  contaminate the next.
- Nothing writes outside `/kaggle/working`. Your v1 adapter is mounted read-only
  by Kaggle and is only ever read.
- v2 is a **new** adapter in a new folder. It does not replace v1, and v1 stays
  archived at `models/v1_original_2026-08-07/` with checksums.
