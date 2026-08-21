# Stages 5 & 6 — the shared machinery, Model A, and Model B

```bash
python -m modeling.run_model_a                  # Model A, 3 seeds, full data
python -m modeling.run_model_b                  # Model B, 3 seeds, full data
python -m modeling.run_model_{a,b} --sample     # smoke test, seconds
python -m modeling.run_model_{a,b} --no-pos-weight   # the ablation
python -m pytest modeling/ -q                   # 47 tests, ~5s
```

> A tuned variant of Model B — dropout, mean+max pooling, cosine learning-rate
> decay — was built and run after Stage 6 was reported. **It lost**: AUPRC
> 0.6022 against 0.6767, because adding a max-pooling channel raised
> scenario-1 recall by 6.6 points while costing 11.0 points of scenario 2,
> which is 88% of all positives. The code has been reverted; the measurement
> and its explanation are kept in `STAGE6_TUNING_RESULT.md`, because the trade
> is itself the argument for average pooling.

This package builds **everything both models share** and runs both of them
through it. Stage 6 added one class to `models.py` and one new
`run_model_b.py`, and changed nothing else — `data.py`, `train.py`,
`runner.py` and `metrics.py` are byte-for-byte as Stage 5 left them. That is
the design goal, not a nice-to-have: the experiment is

> two small networks on **identical data with identical settings**. Model A
> sees **one day**. Model B sees the **28 days ending on that day** and
> predicts the same label. **One variable changes: temporal context.**

If Stage 6 had to edit `data.py`, `train.py` or `runner.py`, the two models
would not have been trained identically and the comparison would be weaker for
it. It did not.

---

## The files

| File | What it owns |
|---|---|
| `data.py` | load → clip → row mask → split by user → fit standardiser on train → apply → tensors |
| `models.py` | `ModelA` (18→32→1 MLP, 641 params) and `ModelB` (1D CNN, 4,897 params) |
| `metrics.py` | AUPRC, AUROC, P@k, per-scenario recall, threshold sweep, the collapse detector |
| `train.py` | the training loop — takes a model object and knows nothing about it |
| `runner.py` | three seeds, checkpoints, metrics JSON, curves, mean ± sd |
| `run_model_a.py` | CLI: argument parsing and the two Model-A-specific lines |
| `run_model_b.py` | CLI: the same, with `ModelB` and `window_view` — plus the patience watch |
| `test_modeling.py` | 28 tests (Stage 5), all aimed at failures that produce no exception |
| `test_model_b.py` | 19 tests (Stage 6), aimed at the one new family: the second axis |

**Stage 6's diff, in full:** `models.py` gained `ModelB` and three named
constants; `run_model_b.py` and `test_model_b.py` are new; this README and
`STAGE6_HANDOVER.md` are documentation. Nothing else in the repository
changed.

The three constants (`MODEL_B_CHANNELS`, `MODEL_B_KERNEL`,
`MODEL_B_DILATION`) live in `models.py` rather than `config.py`, against this
project's convention. The reason is the prime directive: a constant added to
`config.py` is harmless in fact but not provably harmless by inspection, and
"Model A and Model B ran under identical settings" is the claim the whole
project rests on. Move them in Stage 7 if the convention matters more.

`runner.py` is the one file the Stage 5 brief did not name. It exists because
everything around the loop — device resolution, `pos_weight`, the three-seed
sweep, checkpointing, aggregation, plotting — is identical for both models.
Leaving it inside `run_model_a.py` would force Stage 6 to import from a file
named after the other model, or to copy it; a copy is where two runs quietly
stop being identical.

---

## The three rules that make the comparison valid

### 1. One row set, for both models, for training *and* evaluation

Model B cannot score a user's first 27 days — there is no 28-day window
ending there. If Model A scores them and Model B does not, the two models are
measured on different data.

It is worse than untidy. Those rows are the warm-up period: every feature is
exactly 0 because the user has no baseline yet. They are the easiest negatives
in the table, and dropping them lifts the positive rate — which lifts AUPRC
mechanically, handing Model B roughly 6% for nothing.

**The rule: within each user, only rows from their 28th day onward are
scored.** `day_index >= WINDOW_DAYS - 1`, in `data.target_mask`, for both
models and both phases.

Measured on the full r4.2 table:

```
rows 464,687, masked out as warm-up 27,000 (5.8%), positives lost to the mask 0
```

Zero, as predicted — the earliest attack window in r4.2 opens 2010-06-10,
about 160 days into every user's record. `prepare()` measures and logs this
every run rather than trusting it.

The mask selects which rows are **predicted**, not which are **available as
history**. `SplitData` keeps every row of its users in one contiguous array
and carries *positions* into it; `point_view()` gives Model A the target rows,
`window_view()` gives Model B the 28 rows ending at each target. Same array,
same standardiser, one code path.

### 2. Clip before standardising

Z-scores reach 716 against a mean |z| of 0.421 — one input 1,700× the typical
magnitude. Per-feature standardisation does not fix that and can make it
worse: an outlier that extreme inflates the feature's standard deviation and
squashes every ordinary day toward zero.

**Clip to ±10 first**, fixed a priori, identical for train, validation and
test. Measured: `rows with any |z| > 10 before clipping: 2,138 (0.46%)`.

> Beyond ten standard deviations from a person's own baseline, the exact value
> carries no further information — it is equally off the charts.

### 3. Standardise on training rows only

Not a second helping of Stage 3, and the distinction is worth being able to
state:

- **Stage 3** normalised across **time, within a person** — "unusual for them"
- **Stage 5** normalises across **the population, within a feature** — because
  the eighteen features have different typical |z| magnitudes, and unscaled
  inputs hand the first layer a badly-conditioned problem

Fitted on training **target** rows only (warm-up rows are never scored, so
they must not set the scale either), applied to every row everywhere. The mean
and std travel **inside the checkpoint** — a model whose input scaling lives
in another file is not reproducible, and Stage 7 must apply exactly the
transform the weights were trained under.

---

## Model A

```
18 → 32 → 1,  ReLU after the hidden layer,  641 parameters
```

Raw logit out — **no sigmoid in `forward`**. `BCEWithLogitsLoss` applies one
internally; a second one squashes the output twice, gradients shrink and the
model barely trains, with no error and a loss curve that merely looks
disappointing.

No dropout by default. 641 parameters against ~265,000 training rows is not an
overfitting risk. Kept small on purpose: ~1,140 positive training rows against
641 parameters is roughly 1.8 positives per parameter, which is healthy. The
instinct to enlarge the network because the problem is hard is backwards — a
bigger model memorises those 1,140 rows and generalises to nothing.

---

## Model B

```
input                                        (B, 18, 28)
Conv1d(18 → 32, k=3, dilation=1, padding=1)  → ReLU     (B, 32, 28)
Conv1d(32 → 32, k=3, dilation=2, padding=2)  → ReLU     (B, 32, 28)
mean over the time axis                                 (B, 32)
Linear(32 → 1)                                          (B,)  raw logit
```

**4,897 parameters.** Channels-first: 18 features as channels, 28 days as
length, which is what `Conv1d` consumes and what `SplitData.window_view()`
already produces.

**Dilation 2 on the second layer is free reach.** Two stacked k=3 layers see 5
days; dilating the second makes it 7, at identical parameter count. Scenario 2
is a 45–59 day drift, so temporal reach is the quantity that matters, and
buying it with more layers would cost parameters this model cannot afford.

**Average pooling over time, not max.** Max reports the single loudest day —
which is essentially what Model A already sees. Averaging reports the
*sustained level* across the window, which is what a slow build-up produces
and a one-day view cannot. That is the mechanism the experiment predicts will
win on scenario 2, so it is the one wired in.

**Raw logit out, `(B,)` not `(B, 1)`.** No sigmoid, for the same reason as
Model A. And the shape matters: `BCEWithLogitsLoss` handed a `(B, 1)` input
against a `(B,)` target broadcasts to `(B, B)` and trains happily on a loss
that means nothing.

**4,897 parameters against ~1,147 positive training rows** is already a
thinner ratio than Model A's 641 — about 0.23 positives per parameter against
1.8. No dropout, matching Model A. It is deliberately not deeper: a Model B
that wins by memorising those 1,147 rows answers no question anybody asked.

### Documented alternatives, deliberately not implemented

- **Max-pooling over time instead of average** — captures the single loudest
  day rather than the sustained level. Worth reporting as an ablation if time
  allows; average is the choice that matches the hypothesis being tested.
- **Attention pooling** (`Linear(32→1)` → softmax over days → weighted sum),
  ~35 extra parameters. Its appeal is not accuracy but interpretability: it
  produces a **per-day importance map** showing which of the 28 days the model
  actually looked at, which is a far better presentation visual than a single
  score. Only worth adding once the base version is finished and verified.

### The silent failure this architecture invites

A `Conv1d` stack built around the wrong axis convolves across *features* while
treating days as independent channels. It produces numbers. They are
meaningless, and nothing downstream would notice.

Two things guard it. `ModelB.forward` validates the input shape outright —
18 ≠ 28, so a transposed batch raises rather than training. And
`test_model_convolves_across_days_not_across_features` asserts the behavioural
property that distinguishes the two axes: **a pattern shifted eight days later
must score the same**, because convolution with 'same' padding followed by a
global average pool is translation-invariant, while a feature-axis convolution
hands the shifted pattern to entirely different filters.

> The obvious version of that test — rising ramp versus falling ramp, as
> sketched in the Stage 6 brief — was written first and then **discarded,
> because it does not discriminate**: reversing a ramp also permutes the
> channels a feature-axis convolution reads, so the broken model responds too
> and the test passes while the model produces nonsense. `test_model_b.py`
> therefore ships `_FeatureAxisModel`, the bug written out explicitly, and a
> mutation check asserting the real test fails on it. Measured margin: 1e-8
> for the correct model against 1e-2 for the transposed one.

### The prediction, registered before the run

Quoted verbatim from `STAGE6_CONTEXT.md` §3, which was written before any
Stage 6 code existed. It is recorded here so that the result cannot be
retrofitted to it afterwards.

| | prediction |
|---|---|
| **Scenario 2** | **B clearly beats A** — a slow build-up is invisible in one day and obvious in twenty-eight |
| Scenario 1 | roughly a tie — a single day already screams; there is no trend to find |
| Scenario 3 | tie, or B slightly worse; only 12 pooled positives anyway |
| Overall AUPRC | B ahead, but modestly, since scenario 2 dominates the total |

The number to beat, from Stage 5's validation runs:

```
Model A   AUPRC 0.1513 ± 0.0197   (random baseline 0.00429, ~35× chance)
          AUROC 0.9383 ± 0.0141   P@10 0.800 ± 0.141   P@50 0.573 ± 0.115
          scenario 1  49/121 = 40.5%
          scenario 2  225/990 = 22.7%   <- the target
          scenario 3    4/12  = 33.3%
```

**If B ties A overall but clearly wins scenario 2, that is a better result
than a bare "B is better"** — it identifies *which kind of attack needs
history*.

The mechanism has hard pre-training evidence, measured in Stage 3: scenario-2
attack days score only **0.40×** as high in the last third of their window as
in the first third, because the person's own baseline gradually absorbs a
45–59 day attack. Late in a scenario-2 attack, Model A is looking at a day
that has been partly normalised into invisibility. Model B still holds the
loud early days inside its 28-day window.

### One fairness watch-item

All three of Model A's seeds stopped on **patience**, not the epoch cap (best
epochs 16 / 40 / 29; stopped at 23 / 47 / 36). Patience-7 was never binding
against a ceiling.

A 4,897-parameter CNN can converge more slowly than a 641-parameter MLP. If
Model B's best epoch lands at or next to its *last* epoch with validation
AUPRC still trending upward, the budget clipped it — and that biases the
comparison against B, in the direction of the hypothesis being tested.
`run_model_b.py` warns explicitly when that happens.

> **If it fires, raise the budget for BOTH models and re-run Model A too**
> (90 seconds). Raising it for B alone turns the training budget into a second
> variable.

---

## Training

### The class weight — the single most likely way this project fails

```python
BCEWithLogitsLoss(pos_weight = n_negatives / n_positives)
```

Measured on seed 42's training split: **229.7** (263,462 negatives / 1,147
positives). Computed from the *training split*, at train time, every run.
Never hardcoded, never from the whole table — the test set's class balance is
information about the test set, and putting it in the loss function is leakage
that nothing will flag.

At 0.4% positives a batch of 512 holds about **two** positive rows. Without
the weight, ~510 examples say "lower" and 2 say "higher"; the cheapest way to
reduce loss is to output near-zero for everything, and the model converges to
"nobody is an insider" — 99.6% accurate and worthless. **The loss curve looks
perfectly healthy while this happens.** There is no error message.

So every epoch logs five numbers, not one:

```
epoch   train_loss   val_loss   val_AUPRC   val_P@50   mean_predicted_prob
```

plus a distinct-value count. `mean_predicted_prob` flattening to a constant,
or `distinct` collapsing to single digits, *is* the failure.

### Settings

| | |
|---|---|
| batch size | 512 (256 averages under one positive per batch) |
| optimiser | Adam |
| learning rate | 1e-3 |
| max epochs | 50 |
| early stopping | validation **AUPRC**, patience 7 |
| checkpoint kept | the **best-AUPRC** epoch, not the last |

Early stopping is on AUPRC and not validation loss because loss is dominated
by the 99.6% of rows nobody cares about, and can improve for epochs while the
ranking gets worse.

### Device

`config.resolve_device()` prefers MPS and falls back to CPU; `--device
{auto,cpu,mps}` overrides. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set at import.

This is a **portability** decision, not a performance claim. Expect CPU to be
comparable or faster for Model A: at 641 parameters a batch is a few hundred
thousand arithmetic operations, which a CPU core does in microseconds, while
every GPU op costs tens of microseconds just to dispatch. GPUs win on large
matrices, not on many tiny ones. Running on MPS anyway is what *proves* the
pipeline is device-agnostic — a stray `.numpy()`, an accidental float64, or an
op with no Metal kernel all pass silently on CPU. Time both, report the
measurement.

> **Both models must train on the same device.** Model A on CPU and Model B on
> MPS puts a second variable into a two-model comparison — floating-point
> results differ subtly between backends, and a reader can fairly ask whether
> the reported difference is architecture or arithmetic. Resolve once, record
> it, use it for Stages 5, 6 and 7.

### The decision threshold

Swept on **validation**, chosen to maximise F1, **saved inside the
checkpoint**. Never 0.5 — with a weighted loss the outputs are deliberately
not calibrated around it. A model without its threshold is not a working
system, and recomputing the threshold on test data at report time is leakage.

Candidates are quantiles of the observed scores rather than a linear grid over
[0, 1], because a weighted model's outputs cluster and a linear grid spends
most of its budget on empty axis.

---

## Metrics

**AUPRC is the headline and never travels alone** — always printed beside the
random baseline (the positive rate, ~0.0042 on validation). "AUPRC 0.42
against a random baseline of 0.004" is a sentence a reader can evaluate;
"AUPRC 0.42" is a number with no scale.

**AUROC** is reported because the literature uses it, not because it is more
informative here. **P@10 and P@50** are what an analyst with a finite queue
actually experiences.

**Never accuracy**, and never a confusion matrix at 0.5 presented as the
result: predicting "benign" everywhere scores 99.6% and detects nothing.

**Per scenario, counts beside any percentage.** Stage 4 measured that scenario
3 contributes exactly four positive rows to every test split — all ten of its
insiders have precisely two attack days. Recall on four rows has five possible
values. "2 of 4 attack days" is honest; "50%" is a percentage doing an
impression of a measurement. Scenario 2 supplies ~88% of every split's
positives, so any unweighted average across scenarios is a scenario-2 result
wearing a hat.

**Three seeds, mean ± sd.** Stage 4 measured scenario-1 test positives
swinging 32 → 56 across `config.SEEDS` — a 75% range from the split alone,
before any model exists.

---

## Guards against silent failure

Before training:

- both classes present in train and val, counts printed
- `pos_weight` computed from train only, value printed
- no NaN or inf in the tensors
- target rows, positives, and **positives lost to the row mask** — printed
- no user in both train and val
- calendar-day density (a gap would make a "28-day window" span 40 days)

After epoch 1: predicted probability min/max/mean/std, distinct-value count
against a floor, val AUPRC against the random baseline.

After training: best epoch, best val AUPRC, chosen threshold.

> **A validation AUPRC above ~0.95 is a bug, not a triumph.** At 0.4%
> positives with labels this coarse, near-perfect ranking means leakage — most
> likely a split applied by row instead of by user, or a standardiser fitted
> on everything. `run_model_a.py` exits **3** in that case so a scripted
> pipeline stops instead of writing a leaked number into a results table.

---

## Test set discipline

**Stage 5 does not touch the test split. At all.** `prepare()` does not even
build it — the test users are read out of `splits.json` only to be ignored.
Train on train, select on validation, report validation. Test is opened once,
in Stage 7, for both models together, which is also why there will be exactly
one place in this codebase that reads it.

Saving test predictions "just to peek" is how a project ends up with a number
it cannot defend.

---

## Outputs

| Path | |
|---|---|
| `outputs/checkpoints/model_{a,b}_seed{42,123,456}.pt` | weights + threshold + standardiser + config |
| `outputs/model_{a,b}_metrics.json` | per-seed and aggregated **validation** metrics |
| `outputs/figures/model_{a,b}_curves.png` | train/val loss and val AUPRC per epoch |

Identical shape for both models, written by the same `runner.py` code path, so
Stage 7 can load them through one loader. Each checkpoint carries its own
decision threshold and standardiser statistics, plus `input_kind`
(`"point"` / `"window"`) and `window_days` so Stage 7 knows which view to
build without being told.

`--sample` appends `.sample` to every one of them, and `--no-pos-weight` adds
`_nopw`. A smoke run and a real run must never share an output path — in Stage
1 they did once, and a 7-second dev run silently replaced the 1,000-user
table.

---

## Tests

47 tests, ~5 seconds, no r4.2 required (they build a synthetic calendar-dense
frame). Chosen by the criterion Stages 1–4 used: cover the failures that are
**silent**.

### Stage 5 — `test_modeling.py`, 28 tests

1. **Row mask** — a user's first 27 rows are never targets; the 28th is; no
   positives lost; the mask is per-user; history stays readable for Model B;
   a calendar gap is refused.
2. **Standardiser fitted on train only** — a validation split 50× larger and
   shifted by 1,000 must not move the statistics by a millionth.
3. **Clipping** — 716.26 → 10, −34.94 → −10, 0.4 untouched; and clipping
   happens *before* the fit (asserted through the resulting unit std).
4. **`pos_weight`** — equals `n_neg / n_pos` of the training split, changes
   when the split changes, and is not the whole table's ratio.
5. **No sigmoid in the model** — the forward pass returns values outside
   [0, 1]; and a parameter-count tripwire against a future "improvement".
6. **Determinism** — the same seed twice gives bit-identical weights;
   different seeds do not.
7. **User-level integrity** — no user in both train and val tensors;
   overlapping splits are refused; test users never appear.
8. **Collapse detection fires** — a deliberately degenerate model sets the
   flag and warns, end to end through a real training run.

### Stage 6 — `test_model_b.py`, 19 tests

Stage 6 adds one class, so it adds exactly one new *family* of silent
failures: everything to do with the second axis of the input.

1. **The time axis is the one being convolved** — translation invariance,
   plus a mutation check proving the test fails on a feature-axis model, plus
   the direct wiring assertions (`conv1.in_channels == 18`, 'same' padding on
   both layers, receptive field 7) and a refusal of transposed input.
2. **Window contents** — the window ending at target row *t* holds exactly
   rows *t−27 … t*, in date order, no gap, and its **last** row is *t* itself
   (off by one in either direction is invisible: right shape, converges fine,
   and the model is judging the day *before* the day being reported).
   Asserted by encoding user and day index into two features, whose values
   survive clipping and are recovered exactly by inverting the standardiser.
3. **No cross-user contamination** — every row of every window carries the
   target's own user code; and `window_view()` refuses a target sitting inside
   its user's warm-up rather than wrapping to the previous user.
4. **Both models see the same rows** — same count, same order, same labels,
   and `window[:, :, -1]` is exactly Model A's input row. `pos_weight` comes
   from the prepared labels, so it cannot differ between the two views.
5. **No sigmoid** — outputs land outside [0, 1] and no `Sigmoid` module
   exists anywhere in the graph.
6. **Determinism** — same seed twice gives bit-identical weights, seeding in
   the order `runner.run_experiment` uses (before construction, not after).
7. **Shape and size contract** — `(B, 18, 28)` in, `(B,)` out for every batch
   size; a 4,897-parameter tripwire against a future "improvement"; an
   even kernel or a receptive field longer than the window is refused rather
   than silently becoming a no-op.
