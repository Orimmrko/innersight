# InnerSight — Context for Stage 1 (Feature Extraction)

**Attach this file to a new conversation. It contains everything needed to write Stage 1. No prior chat history is required.**

---

## 1. What this project is

A graduation project on **insider threat detection** using the CMU CERT Insider Threat Test Dataset.

**The thesis in one sentence:** an insider does nothing forbidden — they do normal things at abnormal times, in abnormal amounts. So the system never asks *"is this action bad?"* but *"is this unlike **them**?"* Every raw count is converted into a z-score against that specific person's own history, and the models learn from deviations rather than counts.

**Scope has been deliberately cut to the minimum that answers one clean question:** does temporal context help? Two small neural networks are trained on identical data with identical settings; the only difference is that Model A sees one day and Model B sees the preceding 28 days.

**Hard deadline: ~3 days.** Deliverable is working code plus demonstrated training results. Simplicity beats sophistication everywhere.

---

## 2. Decisions already made — treat these as locked

| Decision | Value | Reason |
|---|---|---|
| Dataset | **r4.2 only** | one of only 2 of 10 releases with enough labelled insiders to train on |
| Row granularity | **one row per `(user, day)`** | keeps all 70 insiders; the old 28-day-window design silently deleted 40 of them |
| Features | **18 daily counts**, fixed order | order is the vector layout — never reorder |
| Model A | MLP, input `(18,)` | sees one day |
| Model B | 1D CNN, input `(18, 28)` | sees 28 days, predicts the **same day's** label |
| Split | **by user**, never by day | 600 / 200 / 200, stratified by scenario |
| Loss | `BCEWithLogitsLoss(pos_weight ≈ 293)` | ~0.34% positive rate |
| Metric | **AUPRC** primary, AUROC + P@50 secondary | accuracy is banned |

**Not doing:** graph networks, embeddings, vector databases, web UI, r5.2, cross-version transfer. These were in an earlier over-scoped design and have been removed.

---

## 3. The seven-stage pipeline (Stage 1 is the subject of this brief)

| | Stage | Reads | Writes | Cost |
|---|---|---|---|---|
| **1** | **Extract** | the 5 raw CSVs | `daily_counts.parquet` — user, day, 18 counts | **hours, once** |
| 2 | Label | `answers/insiders.csv` | adds `label`, `scenario` | seconds |
| 3 | Baseline | counts | `dataset.parquet` — 18 z-scores + label | ~1 min |
| 4 | Split | user list | `splits.json` | seconds |
| 5 | Model A | dataset + splits | `model_a.pt`, curves | minutes |
| 6 | Model B | the same two files | `model_b.pt`, curves | minutes |
| 7 | Evaluate | both models | comparison table, plots | seconds |

**Governing rule:** every stage writes a file; the next stage reads that file; nothing recomputes. Stage 1 is the only slow step, and Stages 5–7 will be run dozens of times while debugging.

**Stage 1 must NOT do:** labelling, z-scoring, windowing, splitting, or anything model-related. It produces raw counts and stops.

---

## 4. Where the data is (verified paths on this machine)

```
Full r4.2 (16 GB)
  /Volumes/dataset hard drive/dataset/r4.2/
  also mirrored at:
  /Users/michaelkuksov/Developer/innersight_dataset/r4.2 copy/

30-user sample (168 MB) — USE THIS FOR DEVELOPMENT
  /Users/michaelkuksov/Developer/innersight_dataset/r4.2_sample/

Ground truth (all releases)
  /Volumes/dataset hard drive/dataset/answers/

Project code goes in
  /Users/michaelkuksov/Developer/innersight/
```

The sample contains 30 users, of which **9 are insiders** across all three scenarios — ideal for fast iteration, useless for reporting metrics (30% insider rate).

### File sizes, full r4.2

| File | Size | Rows in the 30-user sample |
|---|---|---|
| `logon.csv` | 58 MB | 67,953 |
| `device.csv` | 29 MB | 26,342 |
| `file.csv` | 193 MB | 26,222 |
| `email.csv` | 1.36 GB | 89,577 |
| `http.csv` | **14.5 GB** | 967,806 |

### Verified schemas (r4.2)

```
logon.csv    id, date, user, pc, activity            activity ∈ {Logon, Logoff}
device.csv   id, date, user, pc, activity            activity ∈ {Connect, Disconnect}
file.csv     id, date, user, pc, filename, content
http.csv     id, date, user, pc, url, content
email.csv    id, date, user, pc, to, cc, bcc, from, size, attachments, content
```

Supporting files (not needed for Stage 1):
```
LDAP/YYYY-MM.csv   employee_name, user_id, email, role, business_unit,
                   functional_unit, department, team, supervisor   (18 monthly snapshots)
psychometric.csv   employee_name, user_id, O, C, E, A, N
```

### Constants

- Date format: **`%m/%d/%Y %H:%M:%S`** — e.g. `01/02/2010 06:49:00`
- Internal email domain: **`dtaa.com`**
- Business hours: **07:00–19:00** (from the dataset's own documentation, not tuned)
- Period covered: Jan 2010 – May 2011 (17 months)
- Population: 1,000 users, 70 insiders (30 scenario-1, 30 scenario-2, 10 scenario-3)

---

## 5. What Stage 1 must produce

One table, one row per `(user, day)`, `day` normalised to midnight.

```
user      day         logon_count  logoff_count  ...  cloud_upload_visits
KPC0073   2010-07-05            2             2  ...                    0
KPC0073   2010-07-07            3             3  ...                    1
NGF0157   2010-07-07            2             2  ...                    0
```

- **~500,000 rows** expected on full r4.2, ~30–40 MB
- Recommended format: **parquet** (instant reload, keeps dtypes). CSV acceptable.
- Suggested path: `data/daily_counts.parquet`

### The 18 features — this exact order is the vector layout

| # | Name | Source | Definition |
|---|---|---|---|
| 0 | `logon_count` | logon | rows where `activity == 'Logon'` |
| 1 | `logoff_count` | logon | rows where `activity == 'Logoff'` |
| 2 | `after_hours_logons` | logon | Logon rows with `hour < 7 or hour >= 19` |
| 3 | `weekend_logons` | logon | Logon rows with `dayofweek >= 5` |
| 4 | `unique_pcs_used` | logon | **DISTINCT** `pc` |
| 5 | `usb_connect_count` | device | `activity == 'Connect'` |
| 6 | `usb_disconnect_count` | device | `activity == 'Disconnect'` |
| 7 | `after_hours_usb` | device | any device row outside 07:00–19:00 |
| 8 | `file_count` | file | all rows |
| 9 | `file_to_removable_count` | file | **see note below** |
| 10 | `unique_filenames` | file | **DISTINCT** `filename` |
| 11 | `email_sent_count` | email | all rows |
| 12 | `email_to_external_count` | email | rows with ≥1 recipient outside `dtaa.com`, across `to` + `cc` + `bcc` |
| 13 | `large_attachment_count` | email | **see note below** |
| 14 | `total_email_size` | email | sum of `size` |
| 15 | `http_request_count` | http | all rows |
| 16 | `job_search_visits` | http | `url` matches a job domain (list below) |
| 17 | `cloud_upload_visits` | http | `url` matches an upload/leak domain (list below) |

### Three r4.2-specific notes — measured, not assumed

**#9 `file_to_removable_count`.** r4.2's `file.csv` has **no** `to_removable_media` column. In r3/r4 *every* file event already **is** a copy to removable media. So this feature equals `file_count`. Compute it as such and record the fact; without it, one of the eighteen would be identically zero and the strongest exfiltration signal would vanish. It diverges from `file_count` only on r5+.

**#10 `unique_filenames`.** Measured on r4.2: 200,000 file rows → 200,000 distinct filenames. Every filename appears exactly once, so this is also a duplicate of `file_count` on this release. Compute it correctly anyway (distinct count); just know it will be degenerate here.

**#13 `large_attachment_count`.** The 1 MiB threshold is **never crossed** on r4.2 — the largest email measured is 106,941 bytes (127,096 on r4.1). This feature is identically zero.
→ **Recommended fix:** define it instead as the `attachments` column value (count of attachments, max 9, non-zero on ~25% of rows). That revives the feature. Document the deviation from the original definition.

**Net effect:** r4.2 gives ~15–16 genuinely informative features out of 18. This is a reportable finding, not a defect to hide.

### Domain lists — extracted from the actual data, with observed counts in the 30-user sample

```python
JOB_SEARCH_DOMAINS = [
    "careerbuilder.com",    # 3347
    "monster.com",          # 2927
    "jobhuntersbible.com",  # 2185
    "job-hunt.org",         # 2185
    "indeed.com",           # 2183
    "simplyhired.com",      # 2150
]

CLOUD_UPLOAD_DOMAINS = [
    "wikileaks.org",   #   7   ← the scenario-1 exfiltration signal
    "dropbox.com",     # 114
    "mediafire.com",   # 340
    "4shared.com",     #   1
]
```

**Do NOT include `linkedin.com` in the job list.** It is the single most-visited domain in the whole dataset (40,446 hits in the sample) — background browsing, not signal.

---

## 6. Engineering requirements

### Recommended architecture: five independent passes, then one merge

Process each log separately into its own partial file, then merge on `(user, day)`:

```
logon.csv   → partial_logon.parquet    (features 0–4)     ~1 min
device.csv  → partial_device.parquet   (features 5–7)     ~1 min
file.csv    → partial_file.parquet     (features 8–10)    ~2 min
email.csv   → partial_email.parquet    (features 11–14)   ~10 min
http.csv    → partial_http.parquet     (features 15–17)   HOURS
                        ↓
                merge on (user, day) → daily_counts.parquet
```

**Why this matters for the deadline:** 15 of the 18 features are ready in about 15 minutes. The 14.5 GB `http.csv` pass runs separately, in the background, while Stages 2–7 are built and debugged against the other four logs. A crash in the http pass costs nothing already done.

### Reading rules

- **Chunk large files** — `pd.read_csv(..., chunksize=50_000)`. Never load `http.csv` or `email.csv` whole.
- **Project away `content` at read time** via `usecols`. It is generated filler, feeds no feature, and is most of the on-disk size.
- **Parse dates with an explicit format.** Letting pandas infer across 28 M rows costs minutes.
- Only `http.csv` requires true streaming; the other four fit in memory on a modern Mac.

### Aggregation rules

- **Sums combine across chunks by addition.** 16 of the 18 features are sums.
- **DISTINCT counts do not sum.** Features **#4 `unique_pcs_used`** and **#10 `unique_filenames`** are distinct counts — the same PC seen in two chunks of one user-day is one PC, not two. Emit unique `(user, day, value)` triples per chunk, deduplicate once at the end, then count.
- Conveniently, `http.csv` — the only genuinely huge file — needs **no** distinct counts. Its three features are a row count and two URL tallies, so memory stays constant regardless of row count.
- **The key universe is the UNION of `(user, day)` across all five logs.** A day present only in the web log still produces a row.
- **Missing means ZERO, not NaN.** "No USB events today" is an observation, not absent data. Fill after the merge.
- `day` must be normalised to midnight (date only, no time component).

### External-email detection (#12)

Count a row if **any** recipient across `to`, `cc`, **and** `bcc` is outside `dtaa.com`. Recipients within a field are `;`-separated. An email addressed internally but blind-copied outside must count — that is precisely the exfiltration pattern one scenario is built around.

---

## 7. Acceptance checks

Run these before declaring Stage 1 done.

**On the 30-user sample** (`r4.2_sample/`), the per-log row counts must match:

```
logon   67,953      device  26,342      file    26,222
email   89,577      http   967,806
```

**On the full r4.2 output:**

- ~1,000 distinct users
- date range spans 2010-01-02 → 2011-05-xx
- ~500,000 rows total
- **no NaNs anywhere** — every missing value filled with 0
- `unique_pcs_used` is never larger than `logon_count + logoff_count`
- feature #13 is identically zero (confirming the known r4.2 quirk) — or non-zero if the `attachments` redefinition was used
- features #9 and #10 equal #8 exactly (confirming the known r4.2 duplication)

**One qualitative check worth more than all the above:** print the daily counts for user **`KPC0073`** around 2010-07-05 → 2010-07-16. He is a scenario-1 insider whose *entire* 17-month history contains **16 device events, all inside that 10-day window, and zero before it**. If those rows don't show USB activity appearing from nothing, the extraction is wrong.

---

## 8. Environment and conventions

- macOS, Apple Silicon. Python 3.13, pandas, PyTorch (MPS, no CUDA).
- pandas 3.0 note: text columns have their own `str` dtype, so `series.dtype == 'object'` is **False**. Use `pd.api.types.is_string_dtype()`. Prefer `.to_numpy()` over `.values`.
- Constants (paths, thresholds, domain lists, the 18 feature names) belong in **one config module** — no magic numbers scattered through the code.
- Typed signatures, `pathlib.Path`, `logging` over `print`, explicit errors that state what was expected and what was found.
- Add a `--sample` / `--limit` flag so the script can be run against the 30-user sample or a truncated row count during development.
- Log progress during the http pass (rows processed, elapsed) — it runs for hours and silence is indistinguishable from a hang.

---

## 9. The ask

Write **Stage 1 only**: a script (or small module) that reads the five r4.2 logs and produces `daily_counts.parquet` with the 18 features defined above, following the five-passes-then-merge architecture, the chunking and distinct-count rules, and passing the acceptance checks.

Do not write Stages 2–7. Do not add z-scores, labels, windows, or models.
