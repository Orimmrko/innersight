# `labeling/` — Stage 2

Turns Stage 1's sparse count table into a calendar-complete, labelled one.
Nothing else in the project reads `answers/insiders.csv`.

```
labeling/
├── __init__.py
├── label.py          the module — calendar fill, then labelling
├── test_label.py     16 tests
└── README.md         this file
```

---

## What it does

Reads `../data/daily_counts.parquet` and `answers/insiders.csv`, writes
`../data/labelled_counts.parquet`. Two jobs, in this order.

**1. Calendar fill.** Each user is reindexed onto every calendar day between
their own first and last observed day, inclusive. Inserted days get `0` for all
18 features, in the original dtype.

**2. Labelling.** Two new `int8` columns:

| Column | Grain | Meaning |
|---|---|---|
| `label` | per **day** | is this day inside an attack window (0/1) |
| `user_scenario` | per **user** | 0 = benign, 1/2/3 = which insider scenario |

No rows are dropped and no z-scores are computed. Those are Stage 3.

---

## Running it

All commands from the **project root**, not from inside this folder.

```bash
python -m labeling.label                     # full table, ~4 seconds
python -m labeling.label --sample            # 30-user sample, ~1 second
python -m labeling.label --user KPC0073      # print one user's days + window
python -m pytest labeling/test_label.py      # 16 tests
```

`--sample` reads `daily_counts.sample.parquet` and writes
`labelled_counts.sample.parquet`. The two output paths are separate constants in
`config.py` for a reason: in Stage 1 the sample and full runs shared one path and
a 7-second dev run silently replaced the 1,000-user table.

Overrides for a machine that is not Michael's laptop: `--counts`, `--answers`,
`--out`, or the `INNERSIGHT_ANSWERS_DIR` environment variable.

---

## Measured output (full r4.2, 1,000 users)

```
Rows before fill    330,452
Rows after fill     464,687   (+134,235 zero-filled)
Users               1,000
Date range          2010-01-02 -> 2011-05-17
Positive rows       1,892     (0.407%)
pos_weight          244.6

scenario  insiders  positive days  mean days/insider
1               30            196                6.5
2               30          1,676               55.9
3               10             20                2.0
```

All 1,892 window-days exist as rows and are labelled; every user's row block is
calendar-dense; no NaNs, no duplicate `(user, day)`, no `label=1` on a benign
user.

`pos_weight` is printed for orientation only. **Stage 5 must recompute it from
its own training split** — freezing the whole-table value would leak the test
set's class balance into the loss.

---

## Why the fill is not cosmetic

Stage 1 emits a row only for a day that produced at least one event. 330,452
rows against 1,000 × 501 = 501,000 possible ones; most of the gap is weekends.

* **Model B reads 28 consecutive days.** On a positional table, 28 *rows* can
  span 40 calendar days. The "28-day window" would silently become something
  else, and nothing would raise.
* **A day of no activity is an observation, not missing data.** Scenario 2 is an
  employee who has resigned and stopped coming in — their silence is the
  strongest thing they do. Positional windowing deletes that signal by
  construction rather than failing to learn it.

The fill is **per user**, first-to-last observed day. Expanding everyone to the
global 501-day range would invent 300 days of absence for someone terminated in
month 4 — an absence nothing in the world produced.

It also has to happen *before* labelling: 1,892 of the window days exist only
after the fill. Without it the positives come out at 1,364 — weekdays only.

---

## Why two columns and not one

`label` is per day; `user_scenario` is per user and stays 1/2/3 on all of an
insider's rows including their ordinary ones. Stage 4 stratifies the user-level
split by `user_scenario`; Stage 7 breaks results down by it. It is *not* called
`scenario` because `insiders.csv` has a column of that name meaning the same
thing at a different grain, and one careless merge would leave an ambiguous
column on the table every later stage reads.

---

## Two facts this stage surfaces, both for the write-up

**Scenario 3 has 20 positive days in the entire dataset** — ten insiders × two
days each. After a user-level 60/20/20 split the test set may hold four positive
rows. When scenario 3 performs badly in Stage 7, that number *is* the
explanation, and saying so in advance is stronger than discovering it afterwards.

**Scenario 2 is 89% of all positives** (1,676 of 1,892), so an overall metric is
very nearly a scenario-2 metric. This is why Stage 7 reports per-scenario results
rather than an average — an average hides that two of the three attack types
barely register.

---

## A known imperfection in the labels — documented, not fixed

The published windows mark the **signature event**, not the full behavioural
anomaly.

* `BIH0745`'s labelled window is 65 minutes on 2010-07-13, yet his heaviest
  after-hours USB file-copying (41 copies across three nights in June and July)
  sits **outside** it.
* `KPC0073`'s window opens 2010-07-07 20:05, but he generated USB device events
  on **2010-07-06** — one day before.

So some genuinely anomalous days are labelled benign and the model is penalised
for finding real anomalies. **This is a property of the dataset, not a bug.**
Widening the windows or inferring labels from the raw logs would be fitting the
labels to the model.

A narrower alternative exists — `answers/r4.2-{1,2,3}/<user>.csv` lists each
insider's malicious events individually. It was considered and rejected: it is
even narrower than the windows, it would cut scenario 1 to a handful of days, and
the window rule is what the literature uses.

### The check worth running by hand once

```bash
python -m labeling.label --sample --user KPC0073
```

Expected: `label = 1` on 07-07 → 07-15, `label = 0` on 07-06 despite two USB
connects that night. That one-day gap is the boundary above. If it looks
different, the date comparison is wrong. (07-10 and 07-11 are a weekend that only
exists because of the fill, and both are correctly positive.)

---

## Details that are easy to get wrong

**`insiders.csv` covers all ten CERT releases.** It must be filtered to
`dataset == "4.2"` as a **string**. Read as a float, the filter starts behaving
by luck: this project would label users from r5.2 who do not exist in its data
and miss the 70 who do. `load_insiders` reads the whole file as text and raises
`GroundTruthError` if the filter matches nothing.

**Dates are compared at day granularity, inclusive at both ends.** Windows carry
timestamps and some are very short. `load_insiders` normalises both ends to
midnight at parse time so the rule is literally true of the values, rather than
true only if every later comparison remembers to strip the time.

**Not every release in that file is zero-padded** (`3/6/2010 1:41:56` appears
under r2), which is one more reason to filter before parsing. The 70 r4.2 rows
all use `%m/%d/%Y %H:%M:%S`, i.e. `config.DATE_FORMAT`.

---

## What raises and what only reports

`verify()` raises `LabellingError` on the four things that cannot be true of a
correct table: rows lost by the fill, duplicate `(user, day)`, NaNs, and a
`label=1` row on a `user_scenario=0` user.

Everything else is a line in the report marked `ok` or `CHECK` — including
"every insider has ≥ 1 positive row", which would mean an insider's window fell
outside their observed range. On r4.2 that is never a labelling bug; it means
Stage 1 lost their activity. It must be investigated, not silently accepted, but
a surprising number is something to look at rather than a reason to abort.

---

## Constants added to `config.py` for this stage

`LABELLED_COUNTS_SAMPLE_PATH`, `INSIDERS_FILENAME`, `INSIDERS_COLUMNS`,
`RELEASE_ID`, `INSIDERS_DATE_FORMAT`, `EXPECTED_INSIDER_COUNT`,
`BENIGN_SCENARIO`, `SCENARIOS`, and the function `insiders_csv_path()`.
`SCENARIO_COLUMN` changed value from `"scenario"` to `"user_scenario"` (it had no
consumers yet).
