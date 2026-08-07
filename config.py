"""Every constant InnerSight uses.

No other module in this project defines a path, a threshold or a magic number.
If a number needs to change, it changes here once and every stage sees it.

The one rule worth stating twice: ``FEATURE_NAMES`` is the vector layout.
It is the CNN's channel order and the column order of every saved checkpoint.
Reordering it raises no error -- it silently makes every stored artifact score
the wrong feature.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR: Final[Path] = OUTPUT_DIR / "checkpoints"
FIGURE_DIR: Final[Path] = OUTPUT_DIR / "figures"

# Files each stage writes. The governing rule of the pipeline: every stage
# writes a file, the next stage reads that file, nothing recomputes.
DAILY_COUNTS_PATH: Final[Path] = DATA_DIR / "daily_counts.parquet"
LABELLED_COUNTS_PATH: Final[Path] = DATA_DIR / "labelled_counts.parquet"
DATASET_PATH: Final[Path] = DATA_DIR / "dataset.parquet"
SPLITS_PATH: Final[Path] = DATA_DIR / "splits.json"
MODEL_A_PATH: Final[Path] = CHECKPOINT_DIR / "model_a.pt"
MODEL_B_PATH: Final[Path] = CHECKPOINT_DIR / "model_b.pt"

# --------------------------------------------------------------------------
# Dataset paths (CERT r4.2)
#
# Both external-drive paths contain spaces ("dataset hard drive", "r4.2 copy").
# pathlib.Path handles that correctly; f-string shell commands do not. Never
# interpolate one of these into a shell string.
#
# The external drive is not always mounted, so each location has candidates
# tried in order. Environment variables win, which is what lets the pipeline
# run on a machine that is not Michael's laptop without editing this file.
# --------------------------------------------------------------------------

_ENV_FULL: Final[str] = "INNERSIGHT_DATASET_FULL"
_ENV_SAMPLE: Final[str] = "INNERSIGHT_DATASET_SAMPLE"
_ENV_ANSWERS: Final[str] = "INNERSIGHT_ANSWERS_DIR"

DATASET_FULL: Final[Path] = Path("/Volumes/dataset hard drive/dataset/r4.2")
DATASET_FULL_MIRROR: Final[Path] = Path(
    "/Users/michaelkuksov/Developer/innersight_dataset/r4.2 copy"
)
DATASET_SAMPLE: Final[Path] = Path(
    "/Users/michaelkuksov/Developer/innersight_dataset/r4.2_sample"
)
ANSWERS_DIR: Final[Path] = Path("/Volumes/dataset hard drive/dataset/answers")
ANSWERS_DIR_MIRROR: Final[Path] = Path(
    "/Users/michaelkuksov/Developer/innersight_dataset/answers"
)

# --------------------------------------------------------------------------
# The 18 features, in the one order that matters
# --------------------------------------------------------------------------

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "logon_count",              # 0
    "logoff_count",             # 1
    "after_hours_logons",       # 2
    "weekend_logons",           # 3
    "unique_pcs_used",          # 4
    "usb_connect_count",        # 5
    "usb_disconnect_count",     # 6
    "after_hours_usb",          # 7
    "file_count",               # 8
    "file_to_removable_count",  # 9
    "unique_filenames",         # 10
    "email_sent_count",         # 11
    "email_to_external_count",  # 12
    "large_attachment_count",   # 13
    "total_email_size",         # 14
    "http_request_count",       # 15
    "job_search_visits",        # 16
    "cloud_upload_visits",      # 17
)

N_FEATURES: Final[int] = len(FEATURE_NAMES)
FEATURE_INDEX: Final[dict[str, int]] = {
    name: i for i, name in enumerate(FEATURE_NAMES)
}

# Column names shared by every stage's dataframes.
USER_COLUMN: Final[str] = "user"
DAY_COLUMN: Final[str] = "day"
LABEL_COLUMN: Final[str] = "label"
SCENARIO_COLUMN: Final[str] = "scenario"

# --------------------------------------------------------------------------
# Raw log format
# --------------------------------------------------------------------------

DATE_FORMAT: Final[str] = "%m/%d/%Y %H:%M:%S"  # e.g. 01/02/2010 06:49:00
INTERNAL_DOMAIN: Final[str] = "dtaa.com"

# From the dataset's own documentation, not tuned on the data. A logon at or
# after BUSINESS_HOUR_END, or before BUSINESS_HOUR_START, is "after hours".
BUSINESS_HOUR_START: Final[int] = 7
BUSINESS_HOUR_END: Final[int] = 19

# Rows per pandas chunk when streaming the big logs. http.csv is ~13.5 GB on
# the full release, so it is never read whole.
CHUNK_SIZE: Final[int] = 50_000

# Saturday and Sunday, matching pandas' Monday=0 weekday numbering.
WEEKEND_WEEKDAYS: Final[tuple[int, int]] = (5, 6)


@dataclass(frozen=True, slots=True)
class LogSpec:
    """The contract for one raw CSV: what it is called and what is inside it.

    ``expected_sample_rows`` is the row count in the 30-user sample release and
    is used only as a checksum -- a mismatch means the sample was rebuilt, not
    that anything is broken.
    """

    filename: str
    columns: tuple[str, ...]
    expected_sample_rows: int

    @property
    def name(self) -> str:
        return self.filename.removesuffix(".csv")


# Verified against the real files.
LOG_SPECS: Final[dict[str, LogSpec]] = {
    "logon": LogSpec(
        filename="logon.csv",
        columns=("id", "date", "user", "pc", "activity"),  # Logon | Logoff
        expected_sample_rows=67_953,
    ),
    "device": LogSpec(
        filename="device.csv",
        columns=("id", "date", "user", "pc", "activity"),  # Connect | Disconnect
        expected_sample_rows=26_342,
    ),
    "file": LogSpec(
        filename="file.csv",
        columns=("id", "date", "user", "pc", "filename", "content"),
        expected_sample_rows=26_222,
    ),
    "email": LogSpec(
        filename="email.csv",
        columns=(
            "id", "date", "user", "pc", "to", "cc", "bcc",
            "from", "size", "attachments", "content",
        ),
        expected_sample_rows=89_577,
    ),
    "http": LogSpec(
        filename="http.csv",
        columns=("id", "date", "user", "pc", "url", "content"),
        expected_sample_rows=967_806,
    ),
}

LOG_NAMES: Final[tuple[str, ...]] = tuple(LOG_SPECS)

LOGON_ACTIVITY: Final[str] = "Logon"
LOGOFF_ACTIVITY: Final[str] = "Logoff"
DEVICE_CONNECT: Final[str] = "Connect"
DEVICE_DISCONNECT: Final[str] = "Disconnect"

# An attachment this size or larger counts as "large" (bytes).
LARGE_ATTACHMENT_BYTES: Final[int] = 1_000_000

# --------------------------------------------------------------------------
# Domain lists, read off the actual data
# --------------------------------------------------------------------------

JOB_SEARCH_DOMAINS: Final[tuple[str, ...]] = (
    "careerbuilder.com",
    "monster.com",
    "jobhuntersbible.com",
    "job-hunt.org",
    "indeed.com",
    "simplyhired.com",
)

# linkedin.com is deliberately excluded. It is the most-visited domain in the
# entire dataset (40,446 hits in a 30-user sample), which makes it background
# browsing rather than a job-hunting signal. Including it would drown feature 16.

CLOUD_UPLOAD_DOMAINS: Final[tuple[str, ...]] = (
    "wikileaks.org",
    "dropbox.com",
    "mediafire.com",
    "4shared.com",
)

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------

WINDOW_DAYS: Final[int] = 28       # how many days Model B sees; Model A sees 1
MIN_HISTORY_DAYS: Final[int] = 30  # before a user's z-scores are trusted

# Z-scores are clipped so one absurd day cannot dominate a gradient step.
Z_CLIP: Final[float] = 10.0
# Guard against division by a zero standard deviation for a constant feature.
STD_FLOOR: Final[float] = 1e-6


@dataclass(frozen=True, slots=True)
class SplitFractions:
    """Train/val/test proportions. Split by USER, never by day -- the same
    person's Monday must not train a model that is then tested on their Tuesday.
    """

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"split fractions must sum to 1.0, got {total!r} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )
        for name, value in (
            ("train", self.train), ("val", self.val), ("test", self.test)
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"split fraction {name} must be strictly between 0 and 1, "
                    f"got {value!r}"
                )


SPLIT: Final[SplitFractions] = SplitFractions(train=0.6, val=0.2, test=0.2)

SEEDS: Final[tuple[int, ...]] = (42, 123, 456)

# At 0.34% positives, a batch of 256 averages under one positive per batch.
BATCH_SIZE: Final[int] = 512
LEARNING_RATE: Final[float] = 1e-3
MAX_EPOCHS: Final[int] = 50
EARLY_STOP_PATIENCE: Final[int] = 5

# Both models are tiny (~640 and ~5,000 parameters). CPU is as fast as or
# faster than MPS at this size because kernel-launch overhead dominates.
# Don't spend time debugging MPS.
DEVICE: Final[str] = "cpu"

MIN_PYTHON: Final[tuple[int, int]] = (3, 11)

# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no candidate location for a dataset directory exists."""


def _first_existing(
    label: str, candidates: tuple[Path | None, ...]
) -> Path:
    tried = [c for c in candidates if c is not None]
    for candidate in tried:
        if candidate.is_dir():
            return candidate
    listing = "\n".join(f"  - {p}" for p in tried)
    raise DatasetNotFoundError(
        f"{label} not found. Expected a directory at one of:\n{listing}\n"
        f"If the external drive is unplugged, plug it in, or set the "
        f"environment variable for this location and re-run."
    )


def _env_path(variable: str) -> Path | None:
    raw = os.environ.get(variable)
    return Path(raw).expanduser() if raw else None


def resolve_dataset(sample: bool = True) -> Path:
    """Return the directory holding the five raw CSVs.

    ``sample=True`` gives the 30-user slice that development runs against;
    ``sample=False`` gives the full r4.2 release, preferring the external drive
    and falling back to the local mirror.
    """
    if sample:
        return _first_existing(
            f"Sample dataset (${_ENV_SAMPLE})",
            (_env_path(_ENV_SAMPLE), DATASET_SAMPLE),
        )
    return _first_existing(
        f"Full r4.2 dataset (${_ENV_FULL})",
        (_env_path(_ENV_FULL), DATASET_FULL, DATASET_FULL_MIRROR),
    )


def resolve_answers() -> Path:
    """Return the directory holding the ground-truth answer files."""
    return _first_existing(
        f"Answers directory (${_ENV_ANSWERS})",
        (_env_path(_ENV_ANSWERS), ANSWERS_DIR, ANSWERS_DIR_MIRROR),
    )


def log_path(name: str, sample: bool = True) -> Path:
    """Return the path to one raw log, e.g. ``log_path("http", sample=True)``."""
    if name not in LOG_SPECS:
        raise KeyError(
            f"unknown log {name!r}; expected one of {', '.join(LOG_NAMES)}"
        )
    return resolve_dataset(sample=sample) / LOG_SPECS[name].filename


def external_drive_mounted() -> bool:
    """Whether the full dataset is reachable. Information, not an error --
    development happens on the sample."""
    return DATASET_FULL.is_dir()


def ensure_directories() -> None:
    """Create the generated-output directories if they are missing."""
    for directory in (DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
