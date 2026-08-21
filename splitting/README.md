# Stage 4 — the train / validation / test split

Turns `data/dataset.parquet` into `data/splits.json`: an assignment of every one
of the 1,000 users to exactly one of train, val and test, for each of the three
seeds in `config.SEEDS`. No data is copied — the split is an *index* into the
dataset, not a new table.

```bash
python -m splitting.split                  # the 1,000-user table, ~1 s
python -m splitting.split --sample         # the 30-user sample
python -m pytest splitting/ -q             # 31 tests
```

---

## Why the smallest stage is the dangerous one

This is roughly sixty lines of real logic and it is the only stage that can
invalidate every number in the final report without producing a single error
message.

Each user contributes ~465 rows. All of them describe the same person, scored
against the same personal baseline, reflecting the same habits, the same shift
pattern, the same email volume. Split by *day* and user X's Monday trains the
model that is then tested on his Tuesday — the model has already met him. The
test score measures memorisation, and memorisation scores **very well**. There
is no exception, no NaN, no warning; just an excellent number that means
nothing.

So: assign **users**, and every row follows the user.

---

## The three decisions

### 1. Stratify by `user_scenario`, not by insider/benign

Each of the four groups — benign, s1, s2, s3 — is split 60/20/20 separately and
the results are concatenated:

| group | total | train | val | test |
|---|---|---|---|---|
| benign | 930 | 558 | 186 | 186 |
| scenario 1 | 30 | 18 | 6 | 6 |
| scenario 2 | 30 | 18 | 6 | 6 |
| scenario 3 | **10** | **6** | **2** | **2** |
| **total** | **1,000** | **600** | **200** | **200** |

Look at the scenario-3 row. Two insiders in test, two attack days each — **four
positive rows in the entire test set for that attack type**. A plain
insider/benign stratification could hand test zero of them, at which point
scenario-3 recall is undefined and a third of the results table quietly
disappears.

That number is not a problem to engineer around. It is a fact about r4.2 and it
belongs in the write-up *before* any result is shown.

### 2. Sort before shuffling

The classic silent bug in this stage is shuffling users in whatever order they
fell out of the parquet file or a Python `set`. That order is not a guarantee,
so the same seed produces a different split on a different machine or after an
unrelated upstream change — while Stages 5, 6 and 7 all read this file and
silently stop being comparable.

Three things make the output a pure function of (user set, scenario, seed):

1. every scenario group is sorted lexicographically before it is touched;
2. every group gets its **own** generator, seeded from `(seed, scenario)` —
   drawing all groups from one stream would make the benign assignment depend on
   how many scenario-1 users happened to exist;
3. the emitted user lists are sorted, so the JSON is stable and a diff between
   two runs means something.

`test_the_input_order_cannot_reach_the_shuffle` exists for exactly this and is
the reason the tests build their populations in a non-lexicographic order.

### 3. Three seeds, and all three are reported

Attack windows differ enormously in length — scenario 2 averages 56 days,
scenario 3 exactly 2 — so *which* insiders land in test moves the positive count
substantially. One split is one draw of that lottery.

Seeds 42, 123 and 456 are all generated, all written, and Stages 5–7 report mean
± sd across them. Generating several and keeping the one with the nicest balance
is selection on the test set.

---

## Measured output — full r4.2

```
1,000 users (930 benign, 30 s1, 30 s2, 10 s3), 464,687 rows, 1,892 positives (0.407%)

seed 42    train 280,809 rows / 1,147 pos (0.408%)   val 90,591 / 362   test 93,287 / 383
seed 123   train 278,199 rows / 1,158 pos (0.416%)   val 93,721 / 375   test 92,767 / 359
seed 456   train 277,042 rows / 1,119 pos (0.404%)   val 93,525 / 386   test 94,120 / 387

test positives by scenario   seed 42: s1=40  s2=339  s3=4
                             seed 123: s1=32  s2=323  s3=4
                             seed 456: s1=56  s2=327  s3=4
```

Three things in that block decide what the results table may claim:

- **Overall positive rate is stable** (0.387%–0.416%) — the class balance a
  model sees barely moves between seeds, so `pos_weight` is not a source of
  variance.
- **Scenario 3 gives exactly 4 test positives, every time.** Not luck: all ten
  s3 insiders have precisely 2 attack days, so any two of them sum to 4. A
  per-scenario recall for s3 therefore has five possible values — 0, ¼, ½, ¾, 1
  — and one missed day moves it by 25 points. Report it as "2 insiders, 4 days",
  never as a percentage alone.
- **Scenario 1 test positives swing 32 → 56 (a 75% range).** s1 windows run 1–12
  days, so which six insiders land in test matters. This is the strongest single
  argument for reporting mean ± sd rather than one number.

Scenario 2 supplies ~88% of every split's positives, so any unweighted average
across scenarios is mostly a scenario-2 result wearing a hat.

Row counts differ between users (users do not all span the same calendar), which
is why a 60/20/20 *user* split gives 60.4/19.5/20.1 of the **rows** rather than
exactly 60/20/20. Expected, and reported rather than corrected — correcting it
would mean splitting by row again.

---

## Output format

```json
{
  "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
  "seeds": [42, 123, 456],
  "n_users": 1000,
  "splits": {"42": {"train": ["AAF0535", "..."], "val": ["..."], "test": ["..."]}},
  "summary": {
    "42": {
      "train": {"users": 600, "by_scenario": {"0": 558, "1": 18, "2": 18, "3": 6},
                "rows": 280809, "positive_rows": 1147, "positive_rate": 0.004084,
                "positives_by_scenario": {"1": 123, "2": 1012, "3": 12}}
    }
  }
}
```

Written with `sort_keys=True` and a fixed indent, so a diff between two runs is
readable. `positives_by_scenario` is beyond the brief's sketch on purpose: it is
what Stage 7 needs to say anything honest about per-scenario recall, and
re-deriving it later means reopening a 16 MB table to answer a question this
stage already knew.

## Checks printed on every run

```
ok   no user appears in two splits, in any seed
ok   union of the three splits = all 1,000 users, in any seed
ok   every scenario is divided in the right proportion, in every seed
ok   the same seed run twice produces an identical split
ok   different seeds produce different splits
ok   every split holds at least one insider of every scenario
```

The first five are fatal — a failure raises `SplitError` and no file is written.
The last is a **warning**, because the 30-user sample has two scenario-2
insiders and mathematically cannot put one in all three splits. That is a fact
about the sample, not a broken split, so `--sample` reports it and continues.

The report also flags any split whose positive rate falls outside 0.2%–0.7%.
Flagging is all it does: reshuffling until the numbers look tidy is selection on
the test set by another name.

## What this stage does not do

- No z-scores, no windowing, no standardisation, no models.
- **The 18 feature columns are never read.** Only `user`, `user_scenario` and
  `label` come off the disk.
- No modified copy of `dataset.parquet`. Three copies of a 16 MB file is waste
  and creates three places for the data to drift apart.
- **No `pos_weight`.** Stage 5 recomputes it from its own training split at
  train time; freezing a whole-table value here would leak the test set's class
  balance into the loss function.

## Tests (31)

Chosen by the Stage 1–3 criterion: cover the failures that are *silent*.

| # | Test | The bug it exists for |
|---|---|---|
| 1 | disjointness | a leaked user turns memorisation into recall |
| 2 | completeness | six users vanish from a group of thirty and nobody counts |
| 3 | stratification (incl. s3 in every split) | test gets zero scenario-3 insiders |
| 4 | determinism | the split drifts and Model A/B stop being comparable |
| 5 | order independence (3 tests) | sort-before-shuffle; also group independence |
| 6 | seeds differ | three "different" seeds that are one split |
| 7 | tiny inputs (14 cases) | a group smaller than the split count drops someone |

Plus guards on the input (`user_scenario` must be constant per user, `user` must
be a string column under pandas 3.0) and on the summary arithmetic.
