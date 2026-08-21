# `extraction/` — Stage 1

Everything to do with turning raw CERT r4.2 logs into daily feature counts lives
here. Nothing else in the project reads the raw logs.

```
extraction/
├── __init__.py
├── extract.py          the module — streams the 5 logs, writes daily counts
├── test_extract.py     9 tests
├── README.md           this file
└── salvage/            where this code came from, and the evidence it works
```

---

## What it does

Reads the five activity logs and emits **one row per `(user, day)`** with the 18
counts in `config.FEATURE_NAMES` order:

```
user      day         logon_count  after_hours_logons  usb_connect_count  ...
KPC0073   2010-07-05            2                   0                  0
KPC0073   2010-07-06            2                   1                  2
```

Output goes to `../data/daily_counts.parquet` (full) or
`../data/daily_counts.sample.parquet` (sample). Everything downstream reads that
file and never touches the raw logs again — this is the boundary where 16 GB
becomes ~40 MB.

## What it deliberately does NOT do

No labelling, no z-scores, no windowing, no splitting, no models. Those are
Stages 2–7. Keeping them out means this expensive pass survives unchanged when a
labelling rule or a baseline parameter is revised.

---

## Running it

All commands from the **project root**, not from inside this folder.

```bash
python -m extraction.extract                  # 30-user sample, ~7 seconds
python -m extraction.extract --full           # full r4.2, 1-3 hours
python -m extraction.extract --user KPC0073   # print one user's daily rows
python -m pytest extraction/test_extract.py   # 9 tests
```

`--full` is the one that matters and the one that takes hours; `http.csv` alone
is 14.5 GB. Start it in a terminal and leave it — Stages 2–4 can be built against
the sample table in the meantime.

---

## Known truths about r4.2 that this code encodes

Measured from the release, not assumed. All three are correct behaviour that
diverges on r5+, and all three belong in the write-up rather than being quietly
absorbed:

| Feature | On r4.2 | Why |
|---|---|---|
| `file_to_removable_count` | identical to `file_count` | r4.2's `file.csv` has no removable-media column — every file event already *is* a USB copy |
| `unique_filenames` | identical to `file_count` | CERT generates a fresh random filename per event, so every filename appears exactly once |
| `large_attachment_count` | identically zero | the largest of 2.6 M emails is 141,909 bytes against a ~1 MB threshold |

**So r4.2 carries 15 informative features, not 18.**

## The domain lists

`config.JOB_SEARCH_DOMAINS` and `config.CLOUD_UPLOAD_DOMAINS` were chosen by
measurement. `craigslist.org` and `linkedin.com` are excluded on purpose and the
reasoning is recorded in `config.py` next to the lists — read it before adding
anything.

The short version: nearly every employee in this dataset browses job sites
regularly, so a raw count of job-site visits discriminates almost nothing. The
signal is that an insider visits them **zero times for fifteen months and then
fifty times in eight weeks** — which only becomes visible after the Stage 3
per-user z-score.

---

## `salvage/`

This code was not written from scratch. It was ported from an earlier, abandoned
version of the project (`innersight2`), and `salvage/` is the record of that:

- `salvage/README.md` — the verdict and what was done
- `salvage/evidence/` — the six validation checks, the run logs, the measured domain hits
- `salvage/provenance/` — the unmodified original file it was ported from

Nothing in `salvage/` is imported or executed. It exists so that "what did you
reuse from the previous attempt?" has an answer you can point at rather than
describe.

## The check that matters

If you change anything in `extract.py`, run this before trusting it:

```bash
python -m extraction.extract --user KPC0073
```

KPC0073 is a scenario-1 insider whose entire 17-month history contains **16
device events, all inside a ten-day window in July 2010, and none before it**.
USB counts must be zero everywhere and non-zero on 2010-07-06, 07, 09, 12 and 15.

No schema assertion proves an extraction is correct. This does.
