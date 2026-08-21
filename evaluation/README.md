# Stage 7 — Evaluate & Compare

The stage the project is submitted on. It opens the test split **once, for both
models, and never again**, and produces `STAGE7_REPORT.md`,
`outputs/final_results.json` and two figures.

Stage 7 reads checkpoints and computes numbers. It does **not** change how
anything was trained: `modeling/data.py`, `models.py`, `metrics.py`,
`train.py` and `runner.py` are imported and never edited, so this stage cannot
retroactively alter the experiment it is measuring.

---

## Run order

The order is not negotiable and the code enforces the important part of it.

```bash
# ---- Phase 1: validation only. Finish all of it first. -------------------
python -m evaluation.run_phase1                 # ~2 min   latency, P@10, guards
python -m evaluation.run_b7                     # ~4 min   capacity control (trains)
python -m modeling.run_model_a --no-pos-weight  # ~90 s    the ablation (trains)

# ---- Phase 2: the test split. One pass. ----------------------------------
python -m evaluation.run_phase2                 # ~2 min

# ---- Deliverables --------------------------------------------------------
python -m evaluation.run_stage7                 # seconds
```

Add `--device mps` to the three scoring/training commands: both models were
trained on MPS and inference on another backend moves the last decimal, which
the reproduces-training guard will report.

`--sample` runs every command against the 30-user data in seconds. Its numbers
are meaningless — 30 users at a ~30% insider rate — and the report it produces
is stamped as such.

### Why Phase 1 must finish first

Once the test split is opened, no model may change. Any adjustment made after
seeing test numbers is fitting to the test set, and the reported result stops
meaning anything. So every remaining question is answered on validation, and
only then does Phase 2 run.

`run_phase2` refuses to start until `outputs/phase1_validation.json` exists.
It also appends to `outputs/TEST_SPLIT_OPENED.json` on every opening and
refuses a second one without `--force --reason '...'`, which is recorded. That
is not a lock — it is a note to the person about to do it that the number they
are about to produce is no longer a single-pass test result.

---

## What each module is for

| module | one job |
|---|---|
| `corpus.py` | the row set: load, clip, mask, slice per split, carry calendar days |
| `checkpoints.py` | one loader for both models; refuses `*_v2_*` |
| `scoring.py` | checkpoint + rows → probabilities and metrics, threshold applied not chosen |
| `guards.py` | the brief's §13 list, as checks that run and print their values |
| `latency.py` | first-detection day, recall by position, and the §5 reading |
| `falsepositives.py` | the top-scoring benign rows, inspected and classified (§8) |
| `phase1.py` | the validation analyses, orchestrated |
| `testpass.py` | the single test pass (§9), with its audit trail |
| `report.py` / `figures.py` | the deliverables (§11, §12) |
| `run_*.py` | CLI entry points |

---

## The three things this stage could get wrong silently

### 1. Refitting the standardiser on the rows being scored

The transform lives inside the checkpoint because Stage 5 put it there.
`Corpus.build()` takes a `Standardiser` as a **required argument** — there is
no code path in this package that fits one on an evaluation split.

`Corpus.refit_train_standardiser()` exists solely so `guards.py` can compare
the checkpoint's values against a fresh train-only fit and print both. Its
result is never applied to anything.

### 2. Re-sweeping the threshold

`evaluation/scoring.py` does not import `sweep_threshold`.
`describe_at_threshold()` computes precision, recall and F1 **at a cutoff
handed in** — a description, not a choice — and `evaluate()` is always called
with `choose_threshold=False`.

`guards.threshold_from_checkpoint` then asserts the applied value is
byte-for-byte the stored one, and prints it.

### 3. Model B v2 reaching a table

The tuning attempt lost (validation AUPRC 0.6767 → 0.6022, double the spread)
and by brief §3 it appears nowhere except one discussion paragraph about the
pooling operator, which needs no weights.

`load_checkpoint()` accepts only filenames matching
`model_(a|b)_seed\d+(\.sample)?\.pt`. Every tagged run — `v2`, `nopw`, `w7` —
fails to match. `load_variant_checkpoint()` reaches the controls deliberately
and still refuses `v2` outright.

---

## Detection latency, and why it comes first

Scenario-2 attacks run 45–59 days. For a day at position 30 of such an attack,
Model B's whole 28-day window sits *inside* the attack: it is reading 28
consecutive anomalous days and saying "attack", which is easy. Roughly half of
all scenario-2 positive days are like that.

Nothing there is leakage — the window reaches strictly backwards, holds only
that user's rows, and no label is ever visible to the model. But *"detects the
insider"* and *"notices an attack it already missed is still running"* are
different products, and only the first is worth claiming.

`latency.py` measures which one this is, two ways:

- **First detection** — for each insider, the earliest day that is both
  labelled an attack day and scored at or above the checkpoint's threshold,
  expressed as day *N* of that insider's attack window. Insiders never
  detected are counted separately; they are a result, not missing data.
- **Recall by position** — the same positives bucketed into days 1–7, 8–14,
  15–28 and 29+.

`read_verdict()` applies the decision rule, whose thresholds
(`PERSISTENCE_RATIO`, `FLAT_RATIO`, `ONSET_MEDIAN_DAYS`) are declared as
module constants **before** any number is in, so the conclusion is a
consequence of the measurement rather than a description of it.

Read the result beside Stage 3's finding that scenario-2 z-scores decay to
**0.40×** across a window: the signal is weakest late, so recall that holds up
late is not simply the window filling with loud days.

---

## The capacity control

Model B has 7.6× Model A's parameters, so *"B wins because it is bigger"* is a
live alternative explanation for the entire result.

Convolutions plus global pooling do not care about sequence length, so
`ModelB(days=7)` has **exactly** `ModelB(days=28)`'s parameter count — 4,897
either way, asserted at the top of `run_b7.py` and again in the test suite.
Same architecture, same capacity, less context. Seven is the floor: two
dilated k=3 layers have a 7-day receptive field, and a shorter window would
leave the second convolution looking at padding.

| outcome | conclusion |
|---|---|
| B@7 between A and B@28 | context is the driver — demonstrated, not asserted |
| B@7 ≈ B@28 | capacity was doing the work, and the report must say so |
| B@7 ≈ A | the 28-day span specifically is what matters |

`run_b7.py` calls `modeling.runner.run_experiment` with a different view
selector and edits nothing in `modeling/`. It is a control, never scored on
test, and its checkpoints are tagged `w7` so the headline loader refuses them.

### One wart, deliberately not fixed

`runner.save_checkpoint` stamps every window checkpoint with
`config.WINDOW_DAYS` rather than the view's actual length, so a B@7 checkpoint
claims 28 days on disk. Stage 7 may not edit `runner.py` (brief §14) and the
control is validation-only and never reloaded, so the field is simply not
trusted: `load_variant_checkpoint()` requires the caller to state the window
length and pushes a probe batch through the weights to verify it.

---

## Outputs

| path | what |
|---|---|
| `STAGE7_REPORT.md` | deliverable 1 — the report, per §12 |
| `outputs/final_results.json` | deliverable 2 — every number, machine-readable |
| `outputs/figures/stage7_comparison.png` | deliverable 3 |
| `outputs/figures/stage7_detection_latency.png` | deliverable 4 |
| `outputs/insider_detection.csv` | deliverable 5 — per-insider table |
| `outputs/phase1_validation.json` | Phase 1's measurements |
| `outputs/phase2_test.json` | Phase 2's measurements |
| `outputs/TEST_SPLIT_OPENED.json` | the audit trail |

Every file gains a `.sample` infix under `--sample`, so a smoke run can never
land on top of a real one. That rule exists because in Stage 1 a seven-second
dev run silently replaced the 1,000-user table and nothing complained.

---

## Exit codes

| code | meaning |
|---|---|
| 0 | clean |
| 2 | bad device or an impossible argument |
| 3 | a guard failed, or the report was built from runs where one did |
| 4 | phase ordering or the audit trail refused the run |

A test AUPRC at or above 0.95 fails the leakage guard and exits 3. At a 0.4%
positive rate with labels this coarse, near-perfect ranking means leakage, not
success — most likely a split applied by row instead of by user, a
standardiser fitted on everything, or a label leaking into a feature.

---

## Tests

```bash
python -m pytest evaluation/ -q      # 54 tests, ~3 s
python -m pytest evaluation/ modeling/ -q
```

The suite is written against the ways this stage produces a *number* rather
than an error: a refitted standardiser, a re-swept threshold, an off-by-one in
"day N of the attack", a v2 checkpoint in the comparison table, a test user in
the training split. Each of those has a test that fabricates the mistake and
asserts the guard catches it.
