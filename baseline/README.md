# Stage 3 — the personal baseline

Turns `data/labelled_counts.parquet` into `data/dataset.parquet`: the same
464,687 rows, the same 22 columns in the same order, with each of the 18 raw
counts replaced by a z-score against that same person's own earlier days.

```bash
python -m baseline.baseline                    # full table, ~5 s
python -m baseline.baseline --sample           # 30-user sample, ~1 s
python -m baseline.baseline --user KPC0073     # raw counts beside z-scores
python -m pytest baseline/ -q                  # 16 tests
```

---

## Why this stage is the project

An insider does nothing forbidden. They do ordinary things at odd hours, in odd
volumes, with credentials that are genuinely theirs. There is no rule to write
and no signature to match, and "3 USB connections" is not evidence of anything —
for the field engineer it is Tuesday, and for the accountant who has never
touched a removable drive it is the loudest event in seventeen months.

The only question that works is **"is this unlike _them_?"**, and this stage is
where it gets answered:

```
z[t] = (x[t] − mean of that user's days STRICTLY BEFORE t)
       ÷ max(std of those same days, floor[f])
```

Everything before Stage 3 is plumbing that gets the counts onto a clean
calendar. Everything after it is architecture — two small networks testing
whether 28 days of this signal beat one day of it. The idea being tested lives
here, in one line of arithmetic.

---

## The three decisions

### 1. Score against the past, never including today

If today's value sits inside the mean it is measured against, a spike partially
cancels itself: the baseline drifts toward the very anomaly you are looking for.
On a short history the error is a factor of two, and nothing downstream would
ever notice — the numbers stay finite, plausible, and quietly wrong.

Strictly-prior also makes the statistic **causal by construction**. It cannot
see the future, so it cannot leak one. That is worth more than the correctness
argument: it means no reviewer has to take the pipeline's word for it.

Implemented as `groupby(user).expanding()` followed by `.shift(1)` within the
same user — expanding first, shift second, so the window is "every day before
_t_" for every _t_, including a user's first day, where the shift correctly
yields nothing.

### 2. A per-feature floor under the denominator

A user whose USB count is 0 on all 180 of their days has a prior std of exactly
zero. Without a floor, the day they plug something in divides by nothing.

```
floor[f] = 0.1 × median over users of (that user's own std of feature f)
```

That quantity is _typical within-person day-to-day variability_ — the median
person's own noise level in that feature — deliberately not the population's
spread, which would mostly measure the fact that different people have different
jobs. Scaling by 0.1 bounds |z| at roughly ten typical-persons'-worth of
variation, which turns infinity into a large finite number. Large finite numbers
are the ones worth looking at.

**One measured deviation from the brief, stated loudly.** The median is taken
over users who exercise the feature at all, not over all 1,000. On r4.2 most
employees never do the things an insider's signal is made of:

| feature | users with own-std of exactly 0 |
|---|---|
| `after_hours_usb` | 88.6% |
| `weekend_logons` | 83.7% |
| `after_hours_logons` | 81.0% |
| `file_count`, `unique_filenames` | 73.6% |
| `usb_connect_count` | 73.5% |
| `cloud_upload_visits` | 66.1% |

For eight of the eighteen features the all-users median is therefore **literally
zero**, the floor collapses onto the 1.0 fallback, and "one USB connection on a
count that is normally zero" becomes one sigma. Measured, both ways, on the full
release:

| | all users | active users only |
|---|---|---|
| insiders passing 2σ | 54 / 70 | **66 / 70** |
| — scenario 1 | 22 / 30 | **30 / 30** |
| — scenario 3 | 5 / 10 | **8 / 10** |
| KPC0073's first USB day | z = 2.0 | **z = 20.9** |
| benign days already over 2σ | 11.6% | 13.4% |

Separation improves roughly five-fold while the benign false-positive rate moves
by under two points — which is what distinguishes a better statistic from a
merely looser one. `config.STD_FLOOR_ACTIVE_USERS_ONLY` or the CLI flag
`--floor-all-users` reproduces the weaker variant for the write-up.

The fallback still exists for the genuinely dead case: `large_attachment_count`
is identically zero for every user on r4.2, so its floor is 1.0 and it
contributes exactly nothing forever, rather than exploding the day it moves.

**A known, bounded leak.** `floor[f]` is a population statistic computed over
all 1,000 users, including the ones Stage 4 will place in the test split.
Strictly this is a leak. It is one scalar per feature derived from a median of
~1,000 values, so a single test user contributes about 0.1% of it, and computing
it from training users only would require the Stage 4 split to exist before
Stage 3 — inverting the pipeline. It is an approximation, it is bounded, and it
is stated in the report rather than left to be discovered.

### 3. A warm-up period

A confident z-score built from three days of history is fiction. Each user's
first `MIN_HISTORY_DAYS` (= 30) rows are emitted as exactly 0 — **invisible
rather than wrong**.

This costs nothing on r4.2 and the run proves it rather than assuming it: the
earliest attack window opens 2010-06-10, the data starts 2010-01-02, so every
insider has roughly five months of history before their attack. `verify()`
raises if any `label == 1` row lands inside a warm-up. Measured: **0**.

There is a second, smaller bound underneath it. `MIN_PRIOR_DAYS = 2` — a row
needs at least two earlier days before it has both a mean and a spread. With
zero prior days the formula still returns a number, and that number is `x/floor`:
for a heavy user, enormous, and meaning nothing but "this person exists". At
`MIN_HISTORY_DAYS = 30` this bound never binds and changes not one value in the
output; it exists so that lowering the warm-up later degrades gracefully instead
of injecting 1,000 spurious spikes. It was found by a test, not by reasoning.

---

## Why an expanding window and not an EMA

An exponential moving average with a ~2-week half-life is the more sophisticated
choice and was considered. It was rejected because **scenario 2 attacks last
45–59 days** — long enough that an EMA would substantially absorb the attack
into the baseline it is being measured against, and scenario 2 is 89% of all
positives.

**The expanding window still absorbs, and the report measures it rather than
hiding it.** The `late_ratio` column is the mean daily peak |z| over a window's
last third divided by its first third:

| scenario | median late/early |
|---|---|
| 1 | 0.61 |
| 2 | **0.40** |
| 3 | not measurable (2-day windows) |

A scenario-2 attack's final fortnight scores 60% lower than its opening one,
against the same person's baseline, because those attack days have been
absorbed into it. That is a real property of the method, it is _why detection
latency matters_, and it is a direct argument for Model B: a 28-day context
window still contains the loud early days after the late ones have faded.

`late_ratio` is deliberately `NaN` for windows shorter than 6 days. Scenario 3's
windows are two days long, where "last third over first third" degenerates into
"day two over day one" — a number that would print convincingly and mean
nothing.

---

## The validation gate

Run before any model exists, because no downstream architecture can invent a
separation that is not in the data, and finding that out on the last day is the
one failure mode with no recovery.

```
scenario   insiders   pass (max)   pass (mean)   median lift (max)   median late/early
1              30         30            12              5.24               0.61
2              30         28             0              1.88               0.40
3              10          8             6              7.48                nan
all            70         66            18              2.78               0.49
below threshold on max|z|: CCL0068, GTD0219, MPM0220, XHW0498
```

### Both mean and max, always

Collapsing 18 z-scores into one number is ambiguous, and **the conclusion flips
depending on the choice**:

- the **mean** dilutes a sharp three-feature attack across fifteen features that
  had no reason to move — 18/70 insiders clear 2σ
- the **max** is noisy enough that an ordinary day already clears the bar —
  66/70 insiders clear 2σ, but so do **13.4% of benign employees' days**

Both columns appear side by side everywhere, and every threshold names the
column it applies to. This ambiguity is a genuine finding about the problem, not
a detail to resolve quietly, and the benign exceedance rate is printed next to
the gate so that "66 of 70 pass" can be read for what it is worth.

`lift` is the more honest single number than either absolute: scenario 1 median
5.2×, scenario 3 median 7.5×, scenario 2 median 1.9× — the slow-drift attack is
genuinely harder, exactly as predicted.

### Reading the four failures

`CCL0068` and `XHW0498` are scenario 2 — slow drift, and the two whose drift is
slowest. `GTD0219` and `MPM0220` are scenario 3, whose entire attack is two days
of logging onto somebody else's machine; both score around 1.5σ, above their own
baseline but under the bar. Neither group is a bug, and both are the honest
answer to "which insiders would this method miss".

`pre_attack` averages **exclude warm-up days**. They are exactly 0 by
construction, and letting ~30 structural zeros into a ~150-day pre-attack
average would deflate every baseline and inflate every lift in the table by
about a fifth. The gate should not flatter itself.

---

## The check to run by hand

```bash
python -m baseline.baseline --user KPC0073
```

Scenario 1. His entire 17-month history contains 16 device events, all inside
2010-07-06 → 07-15 and none before, so his `usb_connect_count` prior mean is
exactly 0 with prior std exactly 0 for ~180 days.

```
       day  label  raw_cloud_upload  z_cloud_upload  raw_usb  z_usb        peak_feature  peak_raw  peak_|z|
2010-02-16      0                 0            0.00        0   0.00   job_search_visits        14     47.32
2010-07-06      0                 0            0.00        2  20.92     after_hours_usb         4    107.11
2010-07-07      1                 1           21.82        1   6.71 cloud_upload_visits         1     21.82
2010-07-09      1                 0           −0.07        3  18.24   usb_connect_count         3     18.24
2010-07-12      1                 1           13.67        1   3.57 cloud_upload_visits         1     13.67
2010-07-15      1                 1            9.72        1   3.47 cloud_upload_visits         1      9.72
```

**Two USB connections score +20.9σ, and four of them after hours score +107σ.**
Large, finite, and computed on real data — that number is the thesis of the
project, and it is worth looking at with your own eyes. If it were `inf` or
`NaN` the floor would not be applied; if it were small, the score-before-absorb
rule would be inverted.

Two details in that table earn their place. `2010-07-06` is `label = 0` — the
answer key's window opens on the 7th, so the loudest day of his career is
formally a negative, which is what detection latency looks like from the other
side. And `2010-02-16` shows an ordinary employee hitting 47σ on a job-search
spike five months before anything happened: real evidence that a per-day
threshold on max|z| is not a detector, and that the interesting question is what
a model does with a **sequence** of these.

---

## What this stage does not do

No windowing, no splitting, no models, and no standardisation across the
population. That last one is Stage 5's job, and the two are orthogonal:

- **Stage 3** normalises across **time, within a person** — "unusual for them"
- **Stage 5** normalises across **the population, within a feature** — because
  different features have different typical |z| magnitudes

Both are needed; neither replaces the other. Labels are read for diagnostics and
written through untouched, and `verify()` raises if either ground-truth column
changed during the stage.

Z-scores are written **unclipped**. `config.Z_CLIP` exists for whoever builds
the training tensors, and it is the threshold for the "rows with any |z| > 10"
diagnostic (2,138 rows, 0.46% — if that were huge, the floor would be too
small). Clipping here would destroy the KPC0073 demonstration for no gain.

---

## Tests

Sixteen, chosen by the Stage 1/2 criterion: cover the failures that are
**silent** — no exception, no NaN, just a number that is subtly wrong and
survives to the results table.

1. **Score-before-absorb** — a constant history with one spike; the spike is
   measured against the pre-spike mean, not one that includes it
2. **Zero-variance history** — a finite z on the day a constant feature changes
3. **Warm-up** — the first 30 rows are exactly 0 and row 31 is not; a positive
   day inside a warm-up is a hard failure
4. **No cross-user leakage** — two users at wildly different levels, interleaved
5. **Row order survives an unsorted input** — the stage sorts to compute and
   restores to write
6. **Shape, column order, dtypes and both label columns survive unchanged**
7. **Numerical stability** — values around 1e8 produce no NaN. This is the test
   that fails if anybody rewrites the expanding aggregation as cumulative sums;
   `E[x²] − E[x]²` on ~1e16 terms can come out negative and become NaN under the
   square root
8. **A table that was never z-scored is rejected** — no negative value anywhere
   means the raw counts were written through, which produces a file of exactly
   the right shape that trains exactly as well as noise

Two of these found real bugs on first run: the first row of every user was being
scored as `x/floor` against a nonexistent history, and the `MIN_PRIOR_DAYS`
bound above is the fix.

The delivered `dataset.parquet` was additionally checked against a naive
reference implementation — a plain Python loop written from the specification
rather than from this code — over 16 users including every hand-checked one.
Worst relative deviation: **6 × 10⁻⁸**.
