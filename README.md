# InnerSight

Insider threat detection on the CMU CERT Insider Threat Test Dataset (r4.2).

**The thesis in one line:** an insider does nothing forbidden — they do normal
things at abnormal times, in abnormal amounts. So the system never asks *"is
this action bad?"* but *"is this unlike **them**?"*

Every raw count becomes a z-score against that person's own history, and the
models learn from deviations rather than counts.

**The experiment:** two small neural networks, trained on identical data with
identical settings. Model A sees **one day**. Model B sees the **28 days ending
on that day** and predicts the same label. One variable changed: temporal
context.

## The seven stages

Every stage writes a file; the next stage reads that file; nothing recomputes.

| | Stage | Module | Writes |
|---|---|---|---|
| 0 | Skeleton & environment | `config.py`, `verify_setup.py` | this repo |
| 1 | Extract | `extraction/` | `data/daily_counts.parquet` — user, day, 18 counts |
| 2 | Label | `label.py` | adds `label`, `scenario` |
| 3 | Baseline (z-scores) | `baseline.py` | `data/dataset.parquet` |
| 4 | Split | `split.py` | `data/splits.json` |
| 5 | Model A (MLP) | `models.py`, `train.py` | `outputs/checkpoints/model_a.pt` |
| 6 | Model B (CNN) | `models.py`, `train.py` | `outputs/checkpoints/model_b.pt` |
| 7 | Evaluate | `evaluate.py` | comparison table, curves |

## Setup

One command — creates the virtual environment, installs the pinned
dependencies and runs the acceptance test:

```bash
bash setup.sh
```

Needs Python ≥ 3.11 (3.13 preferred) on `PATH`. `torch` is a large download, so
the first run takes a few minutes.

To re-run the acceptance test later:

```bash
./.venv/bin/python verify_setup.py          # fast
./.venv/bin/python verify_setup.py --full   # also counts every row
```

## Data

Development runs against a 30-user sample; only the final run uses the full
16 GB release. `config.py` resolves both, tries the external drive first and
falls back to the local mirror, and names the paths it tried if none exist.
Override any of them without editing code:

```bash
export INNERSIGHT_DATASET_SAMPLE=/path/to/r4.2_sample
export INNERSIGHT_DATASET_FULL=/path/to/r4.2
export INNERSIGHT_ANSWERS_DIR=/path/to/answers
```

Every entry point takes a `--sample` flag. `data/` and `outputs/` are
gitignored — generated artifacts don't belong in version control, and
`daily_counts.parquet` alone is ~40 MB.

## Conventions

- All constants live in `config.py`. No path, threshold or magic number
  anywhere else.
- `pathlib.Path` over `os.path`; typed signatures; `X | None` over `Optional[X]`.
- `logging` in pipeline modules; `verify_setup.py` may print.
- Errors state what was expected and what was found.
- `FEATURE_NAMES` order is the vector layout — the CNN's channel order and
  every checkpoint's column order. Reordering it raises no error and silently
  invalidates every saved artifact.
- pandas 3.0: text columns have their own `str` dtype, so
  `series.dtype == 'object'` is **False**. Use `pd.api.types.is_string_dtype()`,
  and prefer `.to_numpy()` over `.values`.
- Device is CPU on purpose: both models are tiny (~640 and ~5,000 parameters)
  and kernel-launch overhead makes MPS slower at this size.
