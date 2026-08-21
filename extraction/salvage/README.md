# Stage 1 salvage — findings

Answering the question in `../SALVAGE_STAGE1_CONTEXT.md`: *did an earlier
version of this project already produce the Stage 1 output?*

Run 2026-08-08.

---

## Verdict

```
FOUND ARTIFACT:     none
  Checks A-F        N/A - nothing to validate. See evidence/search_log.md.

FOUND CODE:         innersight2/backend/innersight/features/daily.py
  Reusable - satisfies all five of the brief's correctness questions as written.

VERDICT:  reuse the code
TIME SAVED IF REUSED:  the 14.5 GB pass still has to run (~1-3 h, unavoidable),
                       but ~4-6 h of writing and debugging Stage 1 is saved.
```

**No cached feature table exists.** Not a partial one, not a stale one, not a
sample one. Four folders, four different search strategies, zero `.parquet` /
`.pkl` / `.feather` / `.h5` / `.npy` files of any kind. Your own project notes
already said the full pipeline had never been run; the filesystem agrees.

**The extraction code, however, was worth every minute.** `innersight2`'s
`features/daily.py` is 30 KB of streaming aggregation with 132 tests behind it,
and it independently answers all five of the questions section 6 of the brief
asks — cross-chunk distinct counts, `content` dropped at read time, explicit
date format, union key universe zero-filled, `cc`/`bcc` scanned. It was ported,
not repaired.

---

## What was produced

`extract.py` at the project root — a single self-contained module importing
**only** `config.py`. Labelling, `CertDataset`, the answers loader, graph
construction, embeddings, models, Flask and Qdrant are all gone. Labelling in
particular is deliberately absent: it belongs to `label.py` (Stage 2), so the
expensive pass can be reused when the labelling rule changes.

Verified against the 30-user sample, **7 seconds end to end**:

| | |
|---|---|
| rows | 8,655 |
| users | 30 (the sample — the full release is ~1,000) |
| days | 436, 2010-01-02 → 2011-05-16 |
| columns | `user`, `day`, then the 18 counts in `FEATURE_NAMES` order |
| NaNs | 0 |
| tests | 9 passing, ruff clean |

The KPC0073 behavioural test reproduces exactly: USB zero everywhere, then
non-zero on 2010-07-06, 07, 09, 12 and 15; 8 connects and 8 disconnects across
his entire 17-month history, nothing before July 2010.

---

## Contents of this folder

```
stage1_salvage/
├── README.md                            this file
├── delivered/
│   ├── extract.py                       snapshot - the live copy is ../extract.py
│   └── test_extract.py                  snapshot - the live copy is ../test_extract.py
├── data/
│   └── daily_counts.sample.parquet      snapshot - also at ../data/
├── evidence/
│   ├── search_log.md                    what was searched and what turned up
│   ├── checks_a_to_f.txt                the brief's six checks, run and answered
│   ├── kpc0073_check_e.txt              the behavioural test, with the rows
│   ├── sample_run_report.txt            full stdout of a sample run
│   ├── sample_run_log.txt               the run's log lines
│   ├── domain_hits.csv                  measured hits per job/cloud domain
│   └── config_issues.md                 two open questions for config.py
└── provenance/
    └── innersight2_features_daily.py    the source extract.py was ported from
```

**The files under `delivered/` and `data/` are snapshots, not the live copies.**
`extract.py`, `test_extract.py` and `data/daily_counts.sample.parquet` at the
project root are what the pipeline actually runs. If you edit one, this folder
goes stale — that is fine, it is a record of what the salvage produced on
2026-08-08, not a second source of truth.

`provenance/innersight2_features_daily.py` is the unmodified original, kept for
one reason: if an examiner asks what was reused from the earlier attempt, the
answer should be a file you can point at rather than a claim. Nothing imports
it. It is not part of the pipeline.

---

## The sample table, and what it is not

`data/daily_counts.sample.parquet` is **30 users, not 1,000**. It is useless for
the final results and must never be what Stage 7's table is computed from.

It is genuinely useful for one thing: building Stages 2–7 today, against real
r4.2 rows with real insiders in them, while the full pass has not been run. It
regenerates in 7 seconds, so treat it as disposable:

```bash
python extract.py                                    # sample, ~7 s
python extract.py --full                             # full r4.2, ~1-3 h
python extract.py --user KPC0073                     # look at an insider's days
python -m pytest test_extract.py                     # 9 tests
```

`--full` writes `data/daily_counts.parquet`, which is the real Stage 1 output
every later stage reads.

---

## Three things worth knowing before Stage 2

**r4.2 carries 16 informative features, not 18.** `file.csv` contributes one
number three times — `file_count`, `file_to_removable_count` and
`unique_filenames` are identical, row by row, because every file event is a
removable-media copy and CERT generates a fresh random filename per event. And
`large_attachment_count` is identically zero: the largest of 2.6M emails is
141,909 bytes against a ~1 MB threshold. All three definitions are correct and
come alive on r5+. Say this before an examiner finds it in a feature-importance
plot; it reads as understanding rather than as a bug.

**Two `config.py` questions are open.** An off-by-one in the http sample row
count, and two job-search domains worth ~6,000 sample hits that are not in
`JOB_SEARCH_DOMAINS`. Neither was changed. See `evidence/config_issues.md`.

**The 14.5 GB pass is still ahead of you.** Nothing salvaged avoids it. What was
avoided is writing and debugging the code that performs it.
