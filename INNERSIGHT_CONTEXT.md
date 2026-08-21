# InnerSight — full project context

**Attach this to any new conversation. It describes the finished project end to end. No prior chat history is needed.**

Last verified: 2026-08-10, against the repository itself.

---

## 1. What this is

A computer-science graduation project on **insider threat detection**, using the CMU CERT Insider Threat Test Dataset, release **r4.2**.

**The thesis, in one sentence:**

> An insider does nothing forbidden — they do normal things at abnormal times and in abnormal amounts, with credentials that are genuinely theirs. So the system never asks *"is this action bad?"* but **"is this unlike *them*?"**

Every raw daily count is converted into a **z-score against that person's own history** before any model sees it. "Three USB connections" becomes "+20σ for this person" or "−0.7σ for that one" depending on who did it.

**The experiment the whole project exists to run:**

> Two small neural networks trained on **identical data with identical settings**. Model A sees **one day**. Model B sees the **28 days ending on that day** and predicts the same label. **One variable changes: temporal context.**

---

## 2. Status

| | Stage | Status |
|---|---|---|
| 0 | Skeleton, config, environment | ✅ done |
| 1 | Extract — raw logs → daily counts | ✅ done |
| 2 | Label — calendar fill + answer key | ✅ done |
| 3 | Baseline — per-user z-scores | ✅ done |
| 4 | Split — by user, 3 seeds | ✅ done |
| 5 | Model A (MLP) — trained, 3 seeds | ✅ done |
| 6 | Model B (1D CNN) — trained, 3 seeds | ✅ done |
| 7 | Evaluate & compare | ⬅️ **code written and tested; the final run has NOT been executed** |

**The test split has never been opened.** `outputs/TEST_SPLIT_OPENED.json` does not exist. Every number below is **validation**.

**Git:** one commit (`Stage 0`). Everything since is uncommitted.

---

## 3. The result so far (validation, 3 seeds, MPS)

| | Model A (1 day) | Model B (28 days) |
|---|---|---|
| **AUPRC** | 0.1513 ± 0.0197 | **0.6767 ± 0.0342** |
| vs random baseline 0.00429 | 35× | **158×** |
| AUROC | 0.9383 ± 0.0141 | 0.9929 ± 0.0009 |
| P@10 | 0.800 ± 0.141 | 0.700 ± 0.424 |
| P@50 | 0.573 ± 0.115 | 0.807 ± 0.204 |
| parameters | 641 | 4,897 |

Per seed — A: 0.1237 / 0.1687 / 0.1615. B: 0.7127 / 0.6306 / 0.6868.
**Model B's worst seed beats Model A's best by 3.7×. No overlap.**

Pooled per-scenario recall, as counts:

| Scenario | Character | Model A | Model B |
|---|---|---|---|
| 1 — USB + leak site | sharp 1–12 day burst | 49 / 121 (40.5%) | 74 / 121 (61.2%) |
| 2 — job hunt + theft | slow 45–59 day drift | 225 / 990 (22.7%) | **712 / 990 (71.9%)** |
| 3 — sysadmin keylogger | 2-day burst | 4 / 12 | 4 / 12 |

Total attack days caught: **278 / 1,123 → 790 / 1,123.**

**Model B's AUPRC after one epoch was 0.2541** — already 1.7× Model A's fully-converged 0.1513. The gain is representational, not training length.

### The registered prediction, scored honestly

Written before Model B existed, copied into `modeling/README.md` before the run:

| | Predicted | Measured | |
|---|---|---|---|
| Scenario 2 | B clearly beats A | 22.7% → 71.9% | ✅ held, 3.2× |
| Scenario 1 | roughly a tie | 40.5% → 61.2% | ❌ **missed** — B won by 20.7 pts |
| Scenario 3 | tie | 4/12 → 4/12 | ✅ held exactly |
| Overall | ahead **but modestly** | 0.151 → 0.677 | ⚠️ direction right, magnitude ~4× off |

The scenario-1 miss has an explanation worth giving: **a spike judged against 27 quiet days inside the same window is a stronger signal than the spike alone.** Temporal context helps short bursts too.

---

## 4. The dataset, and why this one

CERT r4.2: **1,000 employees, 17 months (Jan 2010 – May 2011), 70 insiders**, ~16 GB across five activity logs.

**Why r4.2** — counted from the collection's own `answers/insiders.csv`:

| Release | Insiders | Trainable? |
|---|---|---|
| r1 / r2 / r3.1 / r3.2 / r4.1 | 0–3 | no |
| **r4.2** | **70** | **yes** |
| r5.1 | 4 | no |
| **r5.2** | **99** | yes, but 2× the size |
| r6.1 / r6.2 | 5 | no |

Eight of ten releases have too few labelled insiders to train on. The `.1`/`.2` suffix marks the **"dense needle"** variant. **r4.2 is the smallest release in the collection that can be trained on**, which is why it is the field's standard benchmark.

**Volunteer this weakness:** CERT's own README calls r4.2 *"unrealistically high"* in red-team density — 7% insiders versus near-zero in reality.

### The three attack scenarios

| # | Behaviour | Duration | Count |
|---|---|---|---|
| 1 | after-hours USB copying, upload to a leak site, resignation | 1–12 days | 30 |
| 2 | job hunting, then theft on the way out | 45–59 days | 30 |
| 3 | sysadmin puts a keylogger on his supervisor's machine | 2 days | 10 |

**Scenario 2 is 88% of all positive rows**, so any unweighted average is a scenario-2 result wearing a hat.

---

## 5. The pipeline

```
raw CERT logs (16 GB, 5 CSVs)
   │  extraction/        32.8M events → 330,452 rows, 90 seconds
   ▼
daily_counts.parquet          2.8 MB   user, day, 18 raw counts
   │  labeling/          calendar fill + answer key
   ▼
labelled_counts.parquet       3.5 MB   464,687 rows, 1,892 positives (0.407%)
   │  baseline/          per-user causal z-scores
   ▼
dataset.parquet              16 MB     the models' input
   │  splitting/         600/200/200 users × 3 seeds
   ▼
splits.json                  61 KB
   │  modeling/          Model A · Model B · 3 seeds each
   ▼
checkpoints (weights + threshold + standardiser)
   │  evaluation/        validation analyses, then ONE test pass
   ▼
STAGE7_REPORT.md · final_results.json · figures        ← not yet produced
```

**Governing rule:** every stage writes a file; the next stage reads that file; nothing recomputes.

### The 18 features (order is the input-vector layout — never reorder)

```
 0 logon_count              9 file_to_removable_count
 1 logoff_count            10 unique_filenames
 2 after_hours_logons      11 email_sent_count
 3 weekend_logons          12 email_to_external_count
 4 unique_pcs_used         13 large_attachment_count
 5 usb_connect_count       14 total_email_size
 6 usb_disconnect_count    15 http_request_count
 7 after_hours_usb         16 job_search_visits
 8 file_count              17 cloud_upload_visits
```

**Three are degenerate on r4.2, all measured not assumed:** #9 and #10 are identical to #8 (every file event already is a USB copy; every filename is randomly unique), and #13 is identically zero (largest of 2.6M emails is 141,909 bytes against a 1 MiB threshold). **r4.2 carries 15 informative features, not 18.** All three come alive on r5+.

---

## 6. Key design decisions, with the evidence

**Per-day labels, not 28-day windows.** The original design marked a window positive only if ≥50% fell inside an attack. Scenario 1 lasts 1–12 days and scenario 3 lasts 2 — neither can fill 14 of 28. That rule **silently deleted 40 of 70 insiders** and left 125 positives. Per-day labelling keeps all 70 and yields 1,892. **A 15× increase in signal from a labelling rule, before any model existed.**

**Calendar fill.** Each user is reindexed onto every calendar day between their first and last observed day, zero-filled. **528 of the 1,892 positives (28%) exist only after the fill** — days an insider was inside their window but generated no events. A resigned insider who stopped coming in produces exactly that, and absence is the signal.

**Expanding window, not an EMA.** An exponential average with a 2-week half-life would substantially absorb a 45–59 day scenario-2 attack into the baseline it is measured against. It still absorbs: **scenario-2 days score 0.40× as high in the last third of their window as the first.** That decay is both an honest limitation and the mechanism behind Model B's advantage.

**Std floor = 0.1 × median per-user std, over users who exercise the feature.** The specified formula (median over *all* users) gave **exactly zero for 8 of 18 features**, because 73.5% of employees never connect a USB stick — so it was not a scale estimate, it collapsed to a fallback constant. **State the justification in that order** (the degeneracy is a fact about the data distribution, not about the labels); the 54/70 → 66/70 improvement is confirmation reported afterwards.

**Clip to ±10 before standardising.** z-scores reach 716 against a typical 0.42. Standardising does not fix an outlier — it inflates the feature's std, squashing every normal row toward zero while the outlier still towers.

**Domain lists decided by measurement.** Two features depend on which websites count. One test applied uniformly: *does traffic concentrate inside labelled attack windows?* `aol.com/jobs` added (5/0 and 5/0 in/out); `craigslist.org` dropped (17 in, 22 out); `linkedin.com` dropped (40,446 hits, most-visited domain); `mediafire.com` dropped (~334,000 hits, no ground truth). After removing mediafire, insider BIH0745 has **exactly one day in 500 with upload activity — his labelled attack day.**

**Average pooling, not max.** A post-hoc variant with mean+max pooling **lost** (0.677 → 0.602): scenario 1 rose 6.6 points, scenario 2 fell 11.0, and scenario 2 decides the metric. **The pooling operator selects which attack profile the model can detect.**

**1D CNN, not an RNN.** Not about size — a GRU of the same hidden width is 5,025 parameters against the CNN's 4,897. It is about inductive bias: an RNN's final state is dominated by recent steps, and the measured scenario-2 signal is **loudest early and decays**. Convolution plus average pooling weights all 28 days equally and is translation-invariant (verified: a bump at days 8–11 scores the same as at 16–19, separation 1.49e-08).

---

## 7. Experimental validity — how the comparison is protected

**Identical training, verified two ways.** `train.py` takes a model object and inspects nothing about it. Stage 6 added a class and an entry point and touched no shared module. Verified by diff, and corroborated by timestamps — both checkpoint sets predate every later modification to `train.py` / `runner.py`.

**The tripwire.** `pos_weight` computed from training target rows must print **229.7 / 225.3 / 232.1**. The whole-table figure is 244. **If a run prints 244, the 27-day row mask was not applied first**, meaning the two models are on different rows and the comparison is void.

**One row set for both models.** Within each user, only rows from their 28th day onward are scored — training *and* evaluation. Model B cannot score earlier rows, so Model A must not either; those rows are warm-up zeros (easy negatives) and dropping them for one model only would lift its AUPRC ~6% unearned. **Zero positives are lost** (earliest attack opens 2010-06-10, ~160 days into every record).

**Leakage prevented structurally, not by discipline.** `Corpus.build()` takes a standardiser as a **required argument** — no code path in `evaluation/` can fit one on rows it is about to score. `scoring.py` does not import `sweep_threshold`, and `evaluate()` is always called with `choose_threshold=False`. `load_checkpoint()` accepts only `model_(a|b)_seed\d+(\.sample)?\.pt`, so every tagged run (`v2`, `nopw`, `w7`) fails to match.

**Three seeds, because the split is a lottery.** Test scenario-1 positives swing 40 / 32 / 56 across seeds — a 75% range from split assignment alone. **AUPRC may be quoted mean ± sd; recall-at-threshold may not** (Model A's flagged-row count swung 1,168 / 481 / 107). Report per-scenario results pooled, as counts.

---

## 8. The codebase

**10,120 lines of implementation, 3,080 of tests, 158 test functions**, six packages plus config.

```
config.py            643   every constant; nothing else defines one
verify_setup.py      344   environment check
extraction/          881   Stage 1 — the only package that touches raw CSVs
labeling/            714   Stage 2 — the only one that opens the answer key
baseline/            889   Stage 3 — the thesis
splitting/           654   Stage 4 — never touches the 18 features
modeling/          2,450   Stages 5+6 — data, models, train, metrics, runner
evaluation/        4,213   Stage 7 — reads checkpoints, never trains
code learning/             9 HTML docs explaining the code
html_files/               14 HTML docs explaining each stage's decisions
```

**There is no `main.py` — eleven entry points, one per stage:**

```bash
python -m verify_setup            python -m modeling.run_model_a
python -m extraction.extract      python -m modeling.run_model_b
python -m labeling.label          python -m evaluation.run_phase1
python -m baseline.baseline       python -m evaluation.run_b7
python -m splitting.split         python -m evaluation.run_phase2
                                  python -m evaluation.run_stage7
```

Deliberate: Stage 1 is 90 seconds over 16 GB and Stage 4 is one second, so a single `main` would mean re-running the expensive pass on every debug cycle. It also keeps the test-set evaluation behind its own explicit command.

**Every single-file stage has the same internal shape:** errors → the real work → `verify` → analysis → `compute_stats` / `format_report` → `main`. Private helpers sit directly below their caller.

**`evaluation/guards.py`** is the file to open for any integrity question — eight functions each named for what it refuses to allow: `split_disjointness`, `standardiser_from_checkpoint`, `threshold_from_checkpoint`, `identical_rows`, `finite_predictions`, `beats_random`, `not_suspiciously_good`, `reproduces_training`.

---

## 9. Environment

macOS, Apple Silicon. Python 3.13. `.venv/` built by `./setup.sh` from `requirements.txt` — numpy 2.5.1, pandas 3.0.5, torch 2.13.0, scikit-learn, matplotlib, pytest.

**Always activate the venv** — earlier runs used conda base, and the documented versions are the venv's:

```bash
cd ~/Developer/innersight && source .venv/bin/activate
```

Both models trained on **MPS** (Apple GPU), recorded in every checkpoint. **Stages 6 and 7 must use MPS too** — mixing backends puts a second variable into a two-model comparison. The benchmark is honest: CPU was slightly *faster* on the sample (3.18 s vs 5.64 s), because at 641 parameters kernel-launch overhead dominates. MPS was used anyway to prove the pipeline is device-portable.

**Conventions:** typed signatures, `X | None` over `Optional`, `pathlib.Path`, `logging` over `print`, every constant in `config.py`. pandas 3.0: `series.dtype == 'object'` is **False** for text — use `pd.api.types.is_string_dtype()`.

---

## 10. Outstanding

1. **The Stage 7 final run.** Code written and tested on the sample; ~10 minutes of commands:
   ```bash
   python -m pytest evaluation/ modeling/ -q
   python -m evaluation.run_phase1 --device mps     # latency, P@10, guards
   python -m evaluation.run_b7 --device mps         # capacity control
   python -m modeling.run_model_a --no-pos-weight   # the ablation
   # ── read the phase-1 output, THEN ──
   python -m evaluation.run_phase2 --device mps     # THE TEST SPLIT, once
   python -m evaluation.run_stage7                  # report + JSON + figures
   ```
   `run_phase1` prints a guard that must reproduce **A: 0.1237/0.1687/0.1615, B: 0.7127/0.6306/0.6868**. If it doesn't, stop.

2. **Git.** One commit. Everything since is uncommitted.

3. **Cleanup before packaging.** `_to_delete/`, `__pycache__/` (×8), `.pytest_cache/`, `.ruff_cache/`, `.DS_Store`. And **`.venv/` is 1.1 GB of the 1.2 GB project** — exclude it from any archive (`./setup.sh` rebuilds it).

---

## 11. The two open questions Stage 7 must answer

**Onset or persistence?** Scenario-2 attacks run 45–59 days, so for a day at position 30 Model B's whole window sits *inside* the attack — roughly half of scenario-2 positives are like that. Nothing leaks (the window reaches strictly backwards, holds only that user's rows, no label is visible), but *"detects the insider"* and *"notices an attack it already missed"* are different products. **Measure first-detection day per insider.** Flat recall across position buckets ⇒ onset detection. Climbing ⇒ persistence detector, and the claim must change.

**Capacity or context?** Model B has 7.6× the parameters, so "B wins because it's bigger" is live. The control is **Model B on a 7-day window** — identical architecture, **identical parameter count**, less context. If B@7 lands between A and B@28, context is the driver.

---

## 12. Limitations — all measured, all to be volunteered

- The dataset is synthetic and deliberately dense: 7% insiders versus near-zero in reality.
- Only 15 of 18 features are informative on r4.2.
- **The published labels are narrower than the actual anomalous behaviour.** BIH0745's labelled window is 65 minutes, yet his heaviest after-hours USB copying sits outside it; KPC0073's loudest day of all (z = 107 on after-hours USB, 2010-07-06) is formally labelled 0 because the window opens on the 7th. Reported precision is therefore a lower bound.
- **Scenario 3 rests on 4 test rows** — 10 insiders × 2 attack days, 2 insiders per test split. Recall there has five possible values. Write "2 of 4 attack days", never "50%".
- The std floor is a population statistic including future test users — one scalar per feature from a median of ~1,000 values, so a single test user contributes ~0.1%. Bounded and declared.
- Three seeds conflate split variance with initialisation variance. The correct design is 3 initialisations *within* each split (9 runs); scoped out on time.
- The domain lists encode knowledge of the published scenarios — standard for this benchmark, but a real deployment would not have it on day one.
- A **3-seed ensemble is not valid** here: each seed draws a different split, so one seed's model trained on users in another's validation set.

---

## 13. The AUROC / AUPRC finding — put this in the report

Model A scores **AUROC 0.9383**, comparable to recent published work on r4.2 (LAN, Cai et al. 2024, reports AUC 0.9369 — caveat: temporal split, different labelling, so an anchor rather than like-for-like). On the **same predictions** it scores **AUPRC 0.1513**.

> At a 0.4% positive rate, AUROC is dominated by the ease of ranking hundreds of thousands of obvious negatives. We report both and headline the second.

That single comparison justifies the project's whole metric choice, and it is the strongest paragraph available.
