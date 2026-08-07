# InnerSight — Context for Stage 0 (Project Skeleton & Environment)

**Attach this file to a new conversation. The goal is a clean, working, empty project that Stage 1 can be written into immediately — no code that does anything yet, just structure, dependencies and constants.**

---

## 1. What this project is (background, one screen)

A graduation project on **insider threat detection** using the CMU CERT Insider Threat Test Dataset (release **r4.2**).

**Thesis:** an insider does nothing forbidden — they do normal things at abnormal times, in abnormal amounts. So the system never asks *"is this action bad?"* but *"is this unlike **them**?"* Every raw count becomes a z-score against that person's own history, and the models learn from deviations rather than counts.

**The experiment:** two small neural networks, trained on identical data with identical settings. Model A sees **one day**. Model B sees the **28 days ending on that day** and predicts the same label. One variable changed: temporal context.

**Constraint: ~3 days to submission.** Deliverable is working code plus demonstrated training results. Simplicity beats sophistication everywhere. Nothing goes in this project that isn't needed for those two models and their comparison.

### The seven stages

| | Stage | Writes |
|---|---|---|
| 0 | **Skeleton & environment** ← *this brief* | project folder, venv, config |
| 1 | Extract | `daily_counts.parquet` — user, day, 18 counts |
| 2 | Label | adds `label`, `scenario` |
| 3 | Baseline (z-scores) | `dataset.parquet` |
| 4 | Split | `splits.json` |
| 5 | Model A (MLP) | `model_a.pt` |
| 6 | Model B (CNN) | `model_b.pt` |
| 7 | Evaluate | comparison table, curves |

**Governing rule:** every stage writes a file; the next stage reads that file; nothing recomputes.

---

## 2. What Stage 0 must deliver

1. A project folder with the file structure below, every file created (mostly empty stubs).
2. A virtual environment with all dependencies installed and verified.
3. A **`config.py`** containing every constant the project will ever use.
4. A **`verify_setup.py`** that proves the environment and the data paths both work.
5. `.gitignore`, `requirements.txt`, `README.md`, and `git init`.

**Stage 0 writes no logic.** Stub files with a module docstring and `pass` are correct. The point is that Stage 1 can be written straight into `extract.py` with imports that already resolve.

---

## 3. Target structure

Flat, not nested. With eight modules, a `src/` package only adds import friction.

```
innersight/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py            ← ALL constants live here. Nowhere else.
├── verify_setup.py      ← run this to confirm Stage 0 worked
│
├── extract.py           ← stage 1  (stub)
├── label.py             ← stage 2  (stub)
├── baseline.py          ← stage 3  (stub)
├── split.py             ← stage 4  (stub)
├── models.py            ← stages 5–6, both architectures  (stub)
├── train.py             ← stages 5–6, shared training loop  (stub)
├── evaluate.py          ← stage 7  (stub)
│
├── data/                ← generated .parquet files   (gitignored)
├── outputs/
│   ├── checkpoints/     ← .pt files                  (gitignored)
│   └── figures/         ← training curves, plots     (gitignored)
└── html_files/          ← already exists, leave it
```

`data/` and `outputs/` must exist on disk (with `.gitkeep`) but their contents are gitignored — generated artifacts don't belong in version control, and `daily_counts.parquet` alone is ~40 MB.

---

## 4. Environment

- **macOS, Apple Silicon.** Python 3.11+ (3.13 preferred).
- Create a **virtual environment inside the project folder** (`.venv/`), gitignored. Do not install into system Python.
- Print the resolved Python version during setup and fail loudly if it's below 3.11.

### Dependencies — this is the complete list

```
pandas          # dataframes; the whole pipeline
pyarrow         # parquet read/write
numpy
torch           # both models
scikit-learn    # AUPRC, AUROC, stratified splits, metrics
matplotlib      # training curves and figures
tqdm            # progress bar for the multi-hour http.csv pass
```

That is everything. **Explicitly NOT needed:** xgboost, torch-geometric, flask, qdrant-client, transformers, seaborn. An earlier over-scoped design used them; they have been cut.

Pin versions in `requirements.txt` from whatever actually installs, so the environment is reproducible.

### Note on torch and Apple Silicon

The models are tiny (~640 and ~5,000 parameters). **CPU will be as fast as or faster than MPS**, because kernel-launch overhead dominates at this size. Don't spend time debugging MPS. Have `config.py` expose a `DEVICE` constant defaulting to `"cpu"`, with a comment explaining why.

---

## 5. `config.py` — the important part of Stage 0

Every path, threshold, magic number and name list lives here and is imported everywhere else. Use `pathlib.Path`, never string concatenation.

### Paths

```
DATASET_FULL     /Volumes/dataset hard drive/dataset/r4.2
DATASET_SAMPLE   /Users/michaelkuksov/Developer/innersight_dataset/r4.2_sample
ANSWERS_DIR      /Volumes/dataset hard drive/dataset/answers
PROJECT_ROOT     the folder config.py lives in
DATA_DIR         PROJECT_ROOT / "data"
OUTPUT_DIR       PROJECT_ROOT / "outputs"
```

⚠️ **Both dataset paths contain spaces** (`dataset hard drive`). `pathlib.Path` handles this correctly; f-string shell commands do not.

⚠️ **The external drive is not always mounted.** Provide a resolver function that returns the sample path or the full path on request, and raises a clear, readable error naming the missing path if the drive is absent — not a bare `FileNotFoundError` three call frames deep.

A mirror of the full dataset also exists at `/Users/michaelkuksov/Developer/innersight_dataset/r4.2 copy/` — worth including as a fallback.

### The 18 feature names, in this exact order

This order **is** the vector layout — it is the CNN's channel order and every checkpoint's column order. Reordering it doesn't raise an error; it silently makes every saved artifact score the wrong feature.

```
 0  logon_count                 9  file_to_removable_count
 1  logoff_count               10  unique_filenames
 2  after_hours_logons         11  email_sent_count
 3  weekend_logons             12  email_to_external_count
 4  unique_pcs_used            13  large_attachment_count
 5  usb_connect_count          14  total_email_size
 6  usb_disconnect_count       15  http_request_count
 7  after_hours_usb            16  job_search_visits
 8  file_count                 17  cloud_upload_visits
```

### Dataset constants

```
DATE_FORMAT           "%m/%d/%Y %H:%M:%S"     e.g. 01/02/2010 06:49:00
INTERNAL_DOMAIN       "dtaa.com"
BUSINESS_HOUR_START   7                        from the dataset's own docs, not tuned
BUSINESS_HOUR_END     19
CHUNK_SIZE            50_000
```

### Domain lists — extracted from the actual data

```python
JOB_SEARCH_DOMAINS = [
    "careerbuilder.com", "monster.com", "jobhuntersbible.com",
    "job-hunt.org", "indeed.com", "simplyhired.com",
]

CLOUD_UPLOAD_DOMAINS = [
    "wikileaks.org", "dropbox.com", "mediafire.com", "4shared.com",
]
```

Add a comment: **`linkedin.com` is deliberately excluded** — it is the most-visited domain in the entire dataset (40,446 hits in a 30-user sample), so it is background browsing, not a job-hunting signal.

### Modelling constants

```
WINDOW_DAYS        28          how many days Model B sees
MIN_HISTORY_DAYS   30          before a user's z-scores are trusted
SPLIT_TRAIN/VAL/TEST   0.6 / 0.2 / 0.2      by USER, never by day
SEEDS              [42, 123, 456]
BATCH_SIZE         512         at 0.34% positives, 256 averages <1 positive per batch
LEARNING_RATE      1e-3
MAX_EPOCHS         50
DEVICE             "cpu"
```

---

## 6. `verify_setup.py` — the Stage 0 acceptance test

A single script that runs and prints a clear pass/fail per check:

1. Python version ≥ 3.11.
2. Every dependency imports; print each version.
3. `torch` works — build a tiny tensor and multiply it.
4. `PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR` exist (create if missing).
5. **The sample dataset resolves**, and all five CSVs exist there:
   `logon.csv`, `device.csv`, `file.csv`, `email.csv`, `http.csv`.
6. **Read 5 rows from each sample log** and print the column names. Assert the headers match the expected schema below.
7. Report whether the external drive is mounted — as **information, not an error**. Development happens on the sample; the full dataset is only needed for the final run.
8. Round-trip a 3-row DataFrame through `data/_smoketest.parquet` and delete it, proving pyarrow works.

### Expected schemas (verified against the real files)

```
logon.csv    id, date, user, pc, activity            activity ∈ {Logon, Logoff}
device.csv   id, date, user, pc, activity            activity ∈ {Connect, Disconnect}
file.csv     id, date, user, pc, filename, content
http.csv     id, date, user, pc, url, content
email.csv    id, date, user, pc, to, cc, bcc, from, size, attachments, content
```

### Expected sample row counts (30 users, use as a checksum)

```
logon   67,953      device  26,342      file  26,222
email   89,577      http   967,806
```

The script may skip counting `http.csv` rows by default — 150 MB — and do it only behind a `--full` flag.

---

## 7. Conventions to establish now

- Typed function signatures. `X | None` over `Optional[X]`.
- `pathlib.Path` over `os.path`, everywhere.
- `logging` over `print` in pipeline modules; `verify_setup.py` may print.
- Errors state **what was expected and what was found**.
- Frozen dataclasses for anything that is a contract.
- **No path, threshold or magic number outside `config.py`.**
- Every entry point takes a `--sample` flag to run against the 30-user sample. Development happens on the sample; only the final run uses the full 16 GB.
- `pandas 3.0` note: text columns have their own `str` dtype, so `series.dtype == 'object'` is **False**. Use `pd.api.types.is_string_dtype()`. Prefer `.to_numpy()` over `.values`.

### `.gitignore` must cover

```
.venv/    __pycache__/    *.pyc    .DS_Store
data/*    !data/.gitkeep
outputs/checkpoints/*    outputs/figures/*    !**/.gitkeep
```

### `README.md`

Short. What the project is, the seven stages, how to create the venv, how to run `verify_setup.py`, and the one-line thesis. It will grow as stages land.

---

## 8. The ask

Create the project skeleton described above at `/Users/michaelkuksov/Developer/innersight/`:

- the folder structure, with stub modules
- the virtual environment, with the seven dependencies installed and pinned
- a complete `config.py`
- a working `verify_setup.py`
- `.gitignore`, `README.md`, `requirements.txt`, `git init`

Then **run `verify_setup.py`** and show the output.

Write **no pipeline logic** — `extract.py` and its siblings stay as stubs with a docstring. Stage 1 is a separate conversation.
