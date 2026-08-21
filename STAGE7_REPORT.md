# InnerSight — Stage 7: Evaluate & Compare

An insider does nothing forbidden. They do normal things at abnormal times and
in abnormal amounts, with credentials that are genuinely theirs. So this
project never asks *"is this action bad?"* but **"is this unlike *them*?"** —
every raw daily count is a z-score against that person's own history.

Two small networks were trained on **identical rows with identical settings**.
Model A sees **one day**. Model B sees the **28 days ending on that day** and
predicts the same label. Everything else — the clipping, the standardiser, the
loss, the optimiser, the learning rate, the batch size, the early-stopping
rule, the threshold sweep, the seeds and the device — is shared code, verified
byte-identical by diff. **One variable changes: temporal context.** What
follows is the measurement of what that variable is worth.

Seeds [42, 123, 456], reported as mean ± sd. The test split was opened once, for both
models together, using each checkpoint's own stored standardiser and decision
threshold; neither was recomputed on test.

---

## 1. Headline — test split

|  | Model A (1 day) | Model B (28 days) | random |
|---|---|---|---|
| **AUPRC** | 0.1903 ± 0.0300 | 0.6663 ± 0.1277 | 0.00428 |
| vs random | 45× | 155× | 1× |
| AUROC | 0.9474 ± 0.0136 | 0.9957 ± 0.0004 | 0.5 |
| P@10 | 0.867 ± 0.189 | 0.700 ± 0.245 | — |
| P@50 | 0.580 ± 0.086 | 0.767 ± 0.289 | — |
| parameters | 641 | 4,897 | — |
| context | 1 day | 28 days | — |

Per seed — 123: A 0.1972 / B 0.4970 · 42: A 0.1506 / B 0.6964 · 456: A 0.2230 / B 0.8054

Model B's average precision is **3.5× Model A's** on held-out users neither model has seen, at **155× the random baseline of 0.00428**. The two models differ in one respect: how many days of that person's
own recent history each one is allowed to look at.

---

## 2. Per-scenario recall — pooled across seeds, as counts

| Scenario | Model A | Model B |
|---|---|---|
| **1** — sharp 1–12 day burst | 47 of 128 (36.7%) | 90 of 128 (70.3%) |
| **2** — slow 45–59 day drift | 211 of 989 (21.3%) | 649 of 989 (65.6%) |
| **3** — 2-day burst | 4 of 12 (33.3%) | 6 of 12 (50.0%) |

Counts rather than a mean of per-seed rates, and deliberately so. Stage 5
measured Model A's flagged-row count swinging **1,168 / 481 / 107** across the
three seeds while its AUPRC moved only 0.124 → 0.169: F1 maximisation at a
0.4% positive rate sits on a flat, noisy surface, so a single seed's recall is
a statement about the threshold lottery. Scenario 3 contributes
**4** positive rows per split — 12
pooled across the 3 seeds. "4 of
12 attack days" is what that supports; a
percentage on 4 rows has 5 possible values and
is not a measurement.

---

## 3. Detection latency — *what the project may claim*

Scenario-2 attacks run 45–59 days. For a day at position 30 of such an attack,
Model B's whole 28-day window sits *inside* the attack: it is reading 28
consecutive anomalous days. Nothing there is leakage — the window reaches
strictly backwards, holds only that user's rows, and no label is ever visible
— but *"detects the insider"* and *"notices an attack it already missed is
still running"* are different products, and only the first is worth claiming.

**First detection, on test** — day *N* of that insider's attack window:

| Model | Scope | Median day | IQR | Detected | Never caught |
|---|---|---|---|---|---|
| Model A | all | 1 | 1–1 | 39 / 42 | 3 |
| Model A | scenario 1 | 1 | 1–1 | 18 / 18 | 0 |
| Model A | scenario 2 | 1 | 1–2 | 17 / 18 | 1 |
| Model A | scenario 3 | 1 | 1–1 | 4 / 6 | 2 |
| Model B | all | 1 | 1–7 | 39 / 42 | 3 |
| Model B | scenario 1 | 1 | 1–1 | 18 / 18 | 0 |
| Model B | scenario 2 | 8 | 3–18 | 18 / 18 | 0 |
| Model B | scenario 3 | 1 | 1–1 | 3 / 6 | 3 |

**Recall by position within the attack window** (pooled counts, test):

| Position | Model A | Model B |
|---|---|---|
| days 1-7 | 86 / 239 (36.0%) | 105 / 239 (43.9%) |
| days 8-14 | 48 / 153 (31.4%) | 94 / 153 (61.4%) |
| days 15-28 | 55 / 252 (21.8%) | 174 / 252 (69.0%) |
| days 29+ | 73 / 485 (15.1%) | 372 / 485 (76.7%) |

**The reading** — measured on **validation**, where §5 places this analysis,
with the decision rule and its thresholds fixed in `evaluation/latency.py`
before any of these numbers existed:

- **Model A** — onset: detects insider activity at onset — recall is flat across position within the attack and the median first detection falls in the first week (early 34.9% → late 16.7%, 0.48×, median day 1)
- **Model B** — persistence: identifies ongoing exfiltration campaigns; the median first detection must be stated beside any detection claim (early 43.7% → late 83.1%, 1.90×, median day 1)

Read this beside Stage 3's measurement that scenario-2 z-scores decay to
**0.40×** across an attack window. The signal is *weakest* late, so recall
that holds up late is not the window merely filling with loud days.

---

## 4. The capacity control — Model B on a 7-day window

Model B has 7.6× Model A's parameters, so *"B wins because it is bigger"* is a
live alternative explanation for the entire result. Convolutions plus global
pooling do not care about sequence length, so `ModelB(days=7)` has **exactly**
`ModelB(days=28)`'s parameter count. Same architecture, same capacity, less
context. Seven days is the floor: the two dilated k=3 layers have a 7-day
receptive field.

|  | Context | Parameters | Validation AUPRC |
|---|---|---|---|
| Model A | 1 | 641 | 0.1513 ± 0.0197 |
| Model B @ 7 days | 7 | 4,897 | 0.5767 ± 0.0264 |
| Model B @ 28 days | 28 | 4,897 | 0.6767 ± 0.0342 |

B@7 lands **between** Model A and Model B@28, at 81% of the way from one to the other. Since B@7 and B@28 have **identical parameter counts**, the remaining 19% cannot be capacity. **Temporal context is the driver, and the claim is demonstrated rather than asserted.**

This is a control and not a competing model: it was never scored on test.

---

## 5. The `pos_weight` ablation — Model A, validation

At a 0.4% positive rate a batch of 512 holds about two positive rows. Without
a class weight the cheapest way to reduce the loss is to output near-zero for
everything, and the model converges on "nobody is an insider".

| Model A | AUPRC | P@50 | mean predicted prob | distinct predictions |
|---|---|---|---|---|
| weighted BCE | 0.1513 ± 0.0197 | 0.573 ± 0.115 | 0.183672 | 86,253 |
| plain BCE | 0.3200 ± 0.0181 | 0.887 ± 0.074 | 0.004543 | 83,024 |

The collapse guard did not fire, but average precision falls and the predicted probabilities compress toward zero — the weighted loss is what keeps the rare class visible in the ranking.

Not repeated on Model B: thirteen minutes for a second data point on a claim
Model A already evidences.

---

## 6. The AUROC/AUPRC gap, stated as a finding

Model A scores **AUROC 0.9474** and **AUPRC 0.1903** on the *same
predictions*. Both numbers are correct and they describe different things.

AUROC 0.947 sits beside recent published work on r4.2 — LAN (Cai et al.,
2024) reports 0.9369 — while the average precision on those same predictions
is 0.190 against a random baseline of 0.00428. The
comparison is an anchor rather than like-for-like: LAN uses a temporal split
and a different labelling scheme. The point survives the caveat. At a 0.4%
positive rate AUROC is dominated by the ease of ranking hundreds of thousands
of obvious negatives, and a metric that rewards a model for correctly ignoring
the 99.6% of days nobody was ever going to look at does not describe an
analyst's queue. **Average precision does, which is why it is the headline
here and why the random baseline is printed beside it every time.**

---

## 7. The registered prediction, scored honestly

Written into `STAGE6_CONTEXT.md` **before Model B existed**, and scored here
against the test result rather than quietly dropped.

| Prediction | Outcome | Measured |
|---|---|---|
| scenario 2 would improve | ✅ held | 21.3% → 65.6% |
| scenario 1 would roughly tie | ❌ missed | 36.7% → 70.3%, B ahead by 33.6 points |
| scenario 3 would tie | ❌ missed | 4 of 12 → 6 of 12 attack days |
| overall gain would be modest | ⚠️ direction right, magnitude wrong | B is 3.5× A, against the ~1.2× predicted |

The scenario-1 miss is the interesting one, and it has an explanation worth stating: a spike judged against **27 quiet days inside the same window** is a stronger signal than the same spike judged alone. Temporal context helps short bursts too — not only slow drifts — which is not what the hypothesis said.

The hoped-for surgical shape — *B ties overall but wins scenario 2, thereby
identifying which attack profile needs history* — did not happen: B won across the board.

---

## 8. Limitations

Every one of these was measured, not guessed.

- **The dataset is synthetic and deliberately dense.** CERT r4.2 carries 70
  insiders among 1,000 users — a 7% rate that no real organisation has. The
  per-day positive rate (0.4%) is realistic; the per-*person* rate is not.
- **Only 15 of the 18 features are informative on r4.2.**
  `large_attachment_count` is identically zero across the release, and two
  others barely move.
- **The published labels are narrower than the actual anomalous behaviour.**
  Stage 2 measured that the windows mark the *signature event*: BIH0745's
  heaviest after-hours USB copying sits outside his 65-minute labelled
  window, and KPC0073's loudest day of all (z = 107 on after-hours USB) is
  formally labelled 0. On inspection of the highest-scoring "false
  positives": 47 of the top 60 'false positives' are days belonging to known insiders that fall inside or within 28 days of their labelled attack window. Our highest-scoring false positives are, on inspection, genuine anomalous behaviour that the ground-truth windows fail to label; the reported precision is therefore a lower bound.
- **Scenario 3 rests on four test rows per split.** Its standard deviation
  across seeds measures noise on four rows and nothing else.
- **The Stage 3 standard-deviation floor is a population statistic computed
  over all users, test users included.** It is a median of per-user standard
  deviations, so any one held-out user contributes roughly 0.1% of it —
  bounded, disclosed, and not worth re-running the pipeline over.
- **A tuning attempt on Model B lost, and why it lost is the finding.**
  Dropout 0.2 with mean+max pooling and a cosine schedule took AUPRC from
  0.6767 to 0.6022 with double the spread. Mean+max pooling raised scenario-1
  recall by 6.6 points and dropped scenario-2 by 11.0. Scenario 2 is 88% of
  positives, so it decides the AUPRC. A max channel offers the shortcut *"find
  the single worst day"* — the right instinct for a burst, the wrong one for a
  drift whose individual days are unremarkable. **The pooling operator selects
  which attack profile the model can detect.** That variant is excluded from
  every table above by design.

---

## 9. Future work

Three things this project measured itself into wanting. **r5.2 as a
generalisation test** — 2,000 users, all eighteen features exercised, and a
fourth attack scenario, which would separate "the method works" from "the
method works on r4.2". **Three weight initialisations within each split**,
which is the ensemble this project cannot currently claim: the three seeds
here are three different *splits*, so averaging them measures the split
lottery rather than model variance. **Attention pooling in place of the mean**,
which would keep the drift sensitivity the mean gives while producing a
per-day weight map — turning "this person is anomalous" into "these are the
days that made them anomalous", which is what an analyst actually opens the
alert to see.

---

## Appendix A — guards

Every check from the brief's §13 list, as it ran.

| Phase |  | Guard | Detail |
|---|---|---|---|
| 1 (validation) | ok | test users never appear in train or val | disjoint on all 3 seed(s) (seed 42: 200 test users, seed 123: 200 test users, seed 456: 200 test users) |
| 1 (validation) | ok | seed 42: standardiser loaded from checkpoint, not refit | all 2 checkpoint(s) match a train-only refit to 1e-9; mean[0]=-0.024407 std[0]=1.010643, mean[1]=-0.027305 std[1]=1.011278, mean[2]=+0.001523 std[2]=0.431034 |
| 1 (validation) | ok | seed 42: both models scored on identical rows | 85,191 rows, 362 positive (0.4249%), digest 6edbd6db8b33353d |
| 1 (validation) | ok | seed 123: standardiser loaded from checkpoint, not refit | all 2 checkpoint(s) match a train-only refit to 1e-9; mean[0]=-0.024876 std[0]=1.010124, mean[1]=-0.027750 std[1]=1.009862, mean[2]=+0.001766 std[2]=0.418095 |
| 1 (validation) | ok | seed 123: both models scored on identical rows | 88,321 rows, 375 positive (0.4246%), digest 688b5d3fb2a20536 |
| 1 (validation) | ok | seed 456: standardiser loaded from checkpoint, not refit | all 2 checkpoint(s) match a train-only refit to 1e-9; mean[0]=-0.024772 std[0]=1.011061, mean[1]=-0.027402 std[1]=1.011136, mean[2]=+0.001383 std[2]=0.414344 |
| 1 (validation) | ok | seed 456: both models scored on identical rows | 88,125 rows, 386 positive (0.4380%), digest 15f1e4c43ce7801f |
| 1 (validation) | ok | threshold loaded from checkpoint, not re-swept | model_a seed 42: 0.958207 (1,168 rows flagged); model_b seed 42: 0.817646 (444 rows flagged); model_a seed 123: 0.973918 (481 rows flagged); model_b seed 123: 0.991286 (389 rows flagged); model_a seed 456: 0.993927 (107 rows flagged); model_b seed 456: 0.855652 (398 rows flagged) |
| 1 (validation) | ok | no NaN or inf in any prediction | 523,274 predictions, all finite |
| 1 (validation) | ok | AUPRC above the random baseline for every model and seed | worst is model_a seed 42 at 29.1x random |
| 1 (validation) | ok | no model scores AUPRC >= 0.95 (leakage ceiling) | highest is model_b seed 42 at 0.7127 |
| 1 (validation) | ok | re-scoring validation reproduces the training run's AUPRC | model_a seed 42: 0.1237 vs 0.1237 recorded; model_b seed 42: 0.7127 vs 0.7127 recorded; model_a seed 123: 0.1687 vs 0.1687 recorded; model_b seed 123: 0.6306 vs 0.6306 recorded; model_a seed 456: 0.1615 vs 0.1615 recorded; model_b seed 456: 0.6868 vs 0.6868 recorded |
| 2 (test) | ok | test users never appear in train or val | disjoint on all 3 seed(s) (seed 42: 200 test users, seed 123: 200 test users, seed 456: 200 test users) |
| 2 (test) | ok | seed 42: standardiser loaded from checkpoint, not refit | all 2 checkpoint(s) match a train-only refit to 1e-9; mean[0]=-0.024407 std[0]=1.010643, mean[1]=-0.027305 std[1]=1.011278, mean[2]=+0.001523 std[2]=0.431034 |
| 2 (test) | ok | seed 42: both models scored on identical rows | 87,887 rows, 383 positive (0.4358%), digest 3eacde4b53d0ee60 |
| 2 (test) | ok | seed 42: test split matches Stage 4's measured shape | 87,887 rows (expected ~93,000), 383 positives (expected ~376, 0.4358%), scenario-3 positives 4 (expected exactly 4) |
| 2 (test) | ok | seed 123: standardiser loaded from checkpoint, not refit | all 2 checkpoint(s) match a train-only refit to 1e-9; mean[0]=-0.024876 std[0]=1.010124, mean[1]=-0.027750 std[1]=1.009862, mean[2]=+0.001766 std[2]=0.418095 |
| 2 (test) | ok | seed 123: both models scored on identical rows | 87,367 rows, 359 positive (0.4109%), digest 9f4d0b4da0260dcd |
| 2 (test) | ok | seed 123: test split matches Stage 4's measured shape | 87,367 rows (expected ~93,000), 359 positives (expected ~376, 0.4109%), scenario-3 positives 4 (expected exactly 4) |
| 2 (test) | ok | seed 456: standardiser loaded from checkpoint, not refit | all 2 checkpoint(s) match a train-only refit to 1e-9; mean[0]=-0.024772 std[0]=1.011061, mean[1]=-0.027402 std[1]=1.011136, mean[2]=+0.001383 std[2]=0.414344 |
| 2 (test) | ok | seed 456: both models scored on identical rows | 88,720 rows, 387 positive (0.4362%), digest 831e9a4db2194d1e |
| 2 (test) | ok | seed 456: test split matches Stage 4's measured shape | 88,720 rows (expected ~93,000), 387 positives (expected ~376, 0.4362%), scenario-3 positives 4 (expected exactly 4) |
| 2 (test) | ok | threshold loaded from checkpoint, not re-swept | model_a seed 42: 0.958207 (994 rows flagged); model_b seed 42: 0.817646 (520 rows flagged); model_a seed 123: 0.973918 (436 rows flagged); model_b seed 123: 0.991286 (363 rows flagged); model_a seed 456: 0.993927 (71 rows flagged); model_b seed 456: 0.855652 (322 rows flagged) |
| 2 (test) | ok | no NaN or inf in any prediction | 527,948 predictions, all finite |
| 2 (test) | ok | AUPRC above the random baseline for every model and seed | worst is model_a seed 42 at 34.6x random |
| 2 (test) | ok | no model scores AUPRC >= 0.95 (leakage ceiling) | highest is model_b seed 456 at 0.8054 |
