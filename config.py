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
# The 30-user sample gets its OWN filename. Sharing one path would let a
# 7-second sample run silently overwrite the 1,000-user table, and every
# later stage would read it without complaint.
DAILY_COUNTS_SAMPLE_PATH: Final[Path] = DATA_DIR / "daily_counts.sample.parquet"
LABELLED_COUNTS_PATH: Final[Path] = DATA_DIR / "labelled_counts.parquet"
# Same reasoning as above, one stage later: a --sample labelling run must not
# be able to land on top of the 1,000-user labelled table.
LABELLED_COUNTS_SAMPLE_PATH: Final[Path] = DATA_DIR / "labelled_counts.sample.parquet"
DATASET_PATH: Final[Path] = DATA_DIR / "dataset.parquet"
# And once more, one stage later still: a --sample baseline run must not be
# able to land on top of the z-scored 1,000-user table.
DATASET_SAMPLE_PATH: Final[Path] = DATA_DIR / "dataset.sample.parquet"
SPLITS_PATH: Final[Path] = DATA_DIR / "splits.json"
# Fourth time, same reasoning: a --sample split covers 30 users, and Stages 5-7
# reading it while believing they hold 1,000 would train on six people.
SPLITS_SAMPLE_PATH: Final[Path] = DATA_DIR / "splits.sample.json"
# Stage 5+ writes one checkpoint per (model, seed), not one per model: three
# seeds are three independent draws of the user split and each has its own
# weights, its own decision threshold and its own standardiser. Build the
# names with `checkpoint_path()` below rather than formatting strings inline,
# because the `--sample` variant has to land somewhere else entirely and a
# forgotten suffix is exactly how a 7-second smoke run overwrites a real one.
MODEL_A_NAME: Final[str] = "model_a"
MODEL_B_NAME: Final[str] = "model_b"

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
# The two ground-truth columns Stage 2 adds, and they answer different
# questions. LABEL_COLUMN is per DAY -- was this day inside an attack window.
# SCENARIO_COLUMN is per USER and constant across all of that person's rows,
# including their ordinary days, because Stage 4 stratifies the user-level
# split by it and Stage 7 reports per-scenario results.
#
# It is called "user_scenario" and not "scenario" on purpose: insiders.csv has
# a column literally named "scenario" that means the same thing at a different
# grain, and one careless merge would put an ambiguous name on the table every
# later stage reads.
LABEL_COLUMN: Final[str] = "label"
SCENARIO_COLUMN: Final[str] = "user_scenario"
BENIGN_SCENARIO: Final[int] = 0
SCENARIOS: Final[tuple[int, ...]] = (1, 2, 3)

# --------------------------------------------------------------------------
# Raw log format
# --------------------------------------------------------------------------

DATE_FORMAT: Final[str] = "%m/%d/%Y %H:%M:%S"  # e.g. 01/02/2010 06:49:00
INTERNAL_DOMAIN: Final[str] = "dtaa.com"

# --------------------------------------------------------------------------
# Ground truth (answers/insiders.csv)
# --------------------------------------------------------------------------

INSIDERS_FILENAME: Final[str] = "insiders.csv"
INSIDERS_COLUMNS: Final[tuple[str, ...]] = (
    "dataset", "scenario", "details", "user", "start", "end",
)

# insiders.csv covers ALL ten CERT releases in one file, so it must be filtered
# before use. The value is a STRING: read as a float, "4.2" and "4.1" survive
# but "5.2" == 5.2 comparisons start behaving by luck, and r4.2's own rows are
# indistinguishable from nothing at all. Read the column as text and compare to
# this constant.
RELEASE_ID: Final[str] = "4.2"

# The 70 r4.2 windows use the same "%m/%d/%Y %H:%M:%S" stamps as the logs.
# Other releases in the same file are not zero-padded ("3/6/2010 1:41:56"),
# which is one more reason to filter to RELEASE_ID before parsing dates.
INSIDERS_DATE_FORMAT: Final[str] = DATE_FORMAT

# A window is compared at DAY granularity, inclusive at both ends: a window of
# 65 minutes still marks its whole calendar day. See labeling/README.md.
EXPECTED_INSIDER_COUNT: Final[int] = 70  # on the full release; a checksum only

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
        # 967,806 lines in the sample file, of which one is the header.
        expected_sample_rows=967_805,
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
    "aol.com/jobs",
)

# Matching is substring-against-lowercased-URL, so "aol.com/jobs" (a path, not a
# host) works here. It would not under host equality -- keep that in mind if the
# matcher is ever changed.
#
# Two domains are deliberately EXCLUDED, both decided by measurement rather than
# by intuition. For the two scenario-2 insiders in the 30-user sample, hits were
# counted inside vs outside their labelled attack windows:
#
#     domain               AAF0535 in/out   PNL0301 in/out   benign users hitting it
#     monster.com               10 / 0            7 / 0        21 of 28
#     careerbuilder.com         11 / 0           10 / 0        25 of 28
#     aol.com/jobs               5 / 0            5 / 0        16 of 28   <- INCLUDED
#     craigslist.org            17 / 22           3 / 0        24 of 28   <- excluded
#
# craigslist.org is excluded: it is the only candidate where an insider has MORE
# traffic outside the attack window than inside, it is a general classifieds site
# rather than a job board, and including it would add ~3,800 benign hits of noise
# to feature 16. aol.com/jobs has the same in-window-only signature as the six
# domains already listed, so it is included.
#
# linkedin.com is excluded for a different reason: at 40,446 hits it is the
# most-visited domain in the entire dataset, i.e. background browsing rather than
# a job-hunting signal. Including it would drown feature 16.
#
# Worth remembering: nearly every employee in this dataset browses job sites
# regularly, so a raw count of job-site visits discriminates almost nothing. What
# separates AAF0535 is that he visited them zero times for fifteen months and then
# ~50 times in eight weeks. The signal lives in the per-user z-score, not here.

CLOUD_UPLOAD_DOMAINS: Final[tuple[str, ...]] = (
    "wikileaks.org",
    "dropbox.com",
    "4shared.com",
)

# mediafire.com was REMOVED, by measurement. Hit counts on the full release:
#
#     wikileaks.org    ~60-90     <- the scenario-1 exfiltration signature
#     dropbox.com     ~40,000
#     4shared.com      ~1,000
#     mediafire.com  ~334,000     <- removed
#
# mediafire accounted for ~90% of this feature while carrying no ground-truth
# signal on r4.2 -- no labelled attack in the release involves it. In the 30-user
# sample, scenario-1 insider BIH0745 visited mediafire 38 times OUTSIDE his attack
# window and once inside; wikileaks.org appears 7 times in that sample, every one
# by a scenario-1 insider and every one inside their window.
#
# The damage was per-user rather than global, which is worse than it sounds. Since
# Stage 3 scores each person against their own history, an insider who never uses
# mediafire (KPC0073) keeps a baseline of 0 and his wikileaks day produces a large
# z-score; an insider who browses mediafire routinely (BIH0745) already has a noisy
# non-zero baseline, so the same wikileaks visit disappears into it. Keeping
# mediafire made scenario-1 detection depend on unrelated browsing habits.
#
# dropbox.com and 4shared.com are retained deliberately. Neither carries r4.2
# ground truth either, but keeping them means feature 17 remains a general
# "visits to file-upload destinations" measure rather than a lookup for the one
# domain the answer key names. (Dropbox does become the exfiltration channel in
# scenario 5, which exists in r6.)

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------

WINDOW_DAYS: Final[int] = 28       # how many days Model B sees; Model A sees 1
MIN_HISTORY_DAYS: Final[int] = 30  # before a user's z-scores are trusted

# Z-scores are clipped so one absurd day cannot dominate a gradient step.
# Stage 3 does NOT apply this -- it writes the raw z-scores, because the whole
# point of the KPC0073 hand-check is to see the un-clipped number. Whoever
# builds the training tensors decides whether to clip; Z_CLIP is also the
# reporting threshold for the "rows with |z| > 10" diagnostic.
Z_CLIP: Final[float] = 10.0
# Absolute last-resort guard against dividing by a zero standard deviation.
# Stage 3 normally divides by max(std_prior, STD_FLOOR_BY_FEATURE[f]), which is
# a much larger and better-motivated bound; this one only exists so that no
# code path anywhere can reach a literal 0 denominator.
STD_FLOOR: Final[float] = 1e-6

# --------------------------------------------------------------------------
# Stage 3 -- personal baseline (z-scores)
# --------------------------------------------------------------------------

# The per-feature standard-deviation floor is
#
#     floor[f] = STD_FLOOR_MULTIPLIER * median over users of (that user's own
#                std of feature f)
#
# and it is what turns "a user whose USB count was 0 every day for six months
# then became 2" from a division by zero into a large finite z-score. At 0.1 it
# bounds |z| at roughly 10 x one typical person's worth of day-to-day variation
# in that feature, which is a meaningful ceiling rather than an arbitrary one.
STD_FLOOR_MULTIPLIER: Final[float] = 0.1

# The median is taken over users who actually EXERCISE the feature (their own
# std > 0), not over all 1,000. This is a measured correction, not a preference.
#
# On r4.2 most employees never do the things the insider signal is made of:
# 73.5% of users have a usb_connect_count std of exactly 0, 81% for
# after_hours_logons, 73.6% for file_count, 66.1% for cloud_upload_visits. The
# median over ALL users is therefore literally 0 for eight of the eighteen
# features, the floor collapses onto STD_FLOOR_FALLBACK, and a floor of 1.0 on
# a count that is normally 0 means "one USB connection = one sigma".
#
# Measured on the full release, all-users vs active-users medians:
#
#     validation gate passing 2 sigma   54/70  ->  66/70
#       scenario 1                      22/30  ->  30/30
#       scenario 3                       5/10  ->   8/10
#     KPC0073's first USB day             z=2.0 ->  z=20.9
#     benign days already over 2 sigma   11.6%  ->  13.4%   (barely moves)
#
# The separation improves five-fold while the benign false-positive rate is
# almost unchanged, which is what distinguishes a better statistic from a
# looser one. Set this False to reproduce the all-users variant; the CLI flag
# `--floor-all-users` does the same for one run.
STD_FLOOR_ACTIVE_USERS_ONLY: Final[bool] = True

# If a feature is constant for EVERY user its median per-user std is 0 or NaN
# under either rule, and the formula above would floor at 0. On r4.2
# `large_attachment_count` is exactly this case -- it is identically zero. Fall
# back to 1.0, i.e. "one unit of the raw count is one sigma", which keeps such a
# feature at z = 0 forever instead of producing infinities the day it moves.
STD_FLOOR_FALLBACK: Final[float] = 1.0

# The bar the Stage 3 validation gate holds each insider to: a mean daily
# |z| inside the attack window above this is "measurably unlike themselves".
# Two sigma is the conventional line, chosen before looking at the result.
VALIDATION_Z_THRESHOLD: Final[float] = 2.0


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
# Patience is counted in epochs without a new best VALIDATION AUPRC -- not
# without a new best validation loss. Loss is dominated by the 99.6% of rows
# nobody cares about, so it can improve for epochs while ranking gets worse.
EARLY_STOP_PATIENCE: Final[int] = 7

# Model A: 18 -> MODEL_A_HIDDEN -> 1, ReLU between. ~640 parameters against
# ~1,140 positive training rows, i.e. roughly 1.8 positives per parameter.
# Enlarging this because "the problem is hard" is backwards: a bigger net
# memorises those 1,140 rows and generalises to nothing.
MODEL_A_HIDDEN: Final[int] = 32
# No dropout by default -- 640 parameters against 280,000 rows is not an
# overfitting risk, and every extra knob is one more thing to justify. Raise
# it only if validation loss visibly diverges from training loss, and say so.
DROPOUT: Final[float] = 0.0

# "Of the highest-scoring days, how many were real" -- what an analyst with a
# finite queue actually experiences. Reported at each of these k.
PRECISION_AT_K: Final[tuple[int, ...]] = (10, 50)
# The decision threshold is chosen by sweeping this many quantiles of the
# validation scores and keeping the best F1. Quantiles rather than a linear
# 0..1 grid because the outputs of a pos_weight-ed model are not spread over
# [0, 1] at all -- a linear grid puts nearly every candidate in empty space.
THRESHOLD_SWEEP_POINTS: Final[int] = 512
# A validation AUPRC above this is treated as a BUG, not a result. At 0.4%
# positives with labels this coarse, near-perfect ranking means leakage --
# most likely a split applied by row instead of by user, or a standardiser
# fitted on everything. Stage 5 stops and says so rather than celebrating.
LEAKAGE_AUPRC: Final[float] = 0.95
# The collapse detector. If a weighted loss has failed to bite, every row gets
# essentially the same score and the model has learned "nobody is an insider"
# -- which is 99.6% accurate, worthless, and produces a perfectly healthy
# loss curve while it happens. Counting DISTINCT predicted values catches it
# where no error message will.
MIN_DISTINCT_PREDICTIONS: Final[int] = 1_000

# --------------------------------------------------------------------------
# Device
#
# The machine is Apple Silicon, so the accelerator is MPS (Metal). DEVICE
# resolves rather than being hardcoded, so the project also runs unchanged on
# a Linux box or a Mac without Metal.
#
# This is a portability decision, not a performance claim. Expect CPU to be
# comparable or faster for Model A: at ~640 parameters a batch is a few
# hundred thousand arithmetic operations, which a CPU core does in
# microseconds, while every GPU op costs tens of microseconds just to
# dispatch. GPUs win on large matrices, not on many tiny ones. Running on MPS
# anyway is what *proves* the pipeline is device-agnostic -- a stray
# `.numpy()`, an accidental float64, or an op with no Metal kernel all pass
# silently on CPU and fail the moment the code meets a real accelerator.
#
# Both models must train on the same device. If Model A ran on CPU and Model B
# on MPS, a second variable has entered a two-model comparison and a reader can
# fairly ask whether the difference is architecture or arithmetic.
# --------------------------------------------------------------------------

DEVICE_CHOICES: Final[tuple[str, ...]] = ("auto", "cpu", "mps")
DEVICE_PREFERENCE: Final[str] = "auto"
# A few ops have no Metal kernel. Without this they raise instead of falling
# back to CPU. Set at import time so it is in place before torch is used.
MPS_FALLBACK_ENV: Final[str] = "PYTORCH_ENABLE_MPS_FALLBACK"
os.environ.setdefault(MPS_FALLBACK_ENV, "1")

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


def insiders_csv_path() -> Path:
    """Return the path to the ground-truth window file.

    A function rather than a module-level constant because the answers
    directory may live on an external drive that is not mounted at import
    time; resolving eagerly would make importing ``config`` fail on a laptop
    with the drive unplugged.
    """
    return resolve_answers() / INSIDERS_FILENAME


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


# --------------------------------------------------------------------------
# Stage 5+ output paths
#
# One rule, and it has bitten this project once already: a --sample run and a
# full run must never share an output path. In Stage 1 a 7-second dev run
# silently replaced the 1,000-user table and nothing complained. Every
# function below therefore takes `sample` and puts ".sample" in the name.
# --------------------------------------------------------------------------


def _sample_infix(sample: bool) -> str:
    return ".sample" if sample else ""


def checkpoint_path(model: str, seed: int, sample: bool = False) -> Path:
    """Where one trained model's weights, threshold and standardiser live.

    e.g. ``outputs/checkpoints/model_a_seed42.pt``. A checkpoint carries the
    standardiser's mean and std with it: a model whose input scaling lives in
    a different file is not reproducible, and Stage 7 must apply exactly the
    transform the weights were trained under.
    """
    return CHECKPOINT_DIR / f"{model}_seed{seed}{_sample_infix(sample)}.pt"


def metrics_path(model: str, sample: bool = False, tag: str = "") -> Path:
    """Where the per-seed and aggregated validation metrics land.

    ``tag`` names a variant run so the ablation cannot overwrite the real
    result: ``metrics_path("model_a", tag="nopw")`` is a different file.
    """
    suffix = f"_{tag}" if tag else ""
    return OUTPUT_DIR / f"{model}{suffix}_metrics{_sample_infix(sample)}.json"


def figure_path(model: str, sample: bool = False, tag: str = "") -> Path:
    """Where the training-curve figure lands."""
    suffix = f"_{tag}" if tag else ""
    return FIGURE_DIR / f"{model}{suffix}_curves{_sample_infix(sample)}.png"


def resolve_device(preference: str = DEVICE_PREFERENCE) -> str:
    """Return the torch device string to train on.

    ``"auto"`` prefers MPS and falls back to CPU; an explicit ``"mps"`` on a
    machine without Metal is a mistake worth an exception rather than a silent
    downgrade, because the whole point of asking for MPS is to test it.

    torch is imported inside the function: ``config`` is imported by stages
    that have nothing to do with modelling, and importing torch costs a couple
    of seconds every time.
    """
    if preference not in DEVICE_CHOICES:
        raise ValueError(
            f"device must be one of {', '.join(DEVICE_CHOICES)}, "
            f"got {preference!r}"
        )
    if preference == "cpu":
        return "cpu"

    import torch  # noqa: PLC0415  (deliberate: keep torch out of config's import cost)

    available = torch.backends.mps.is_available()
    if preference == "mps":
        if not available:
            raise RuntimeError(
                "device 'mps' was requested but torch.backends.mps."
                "is_available() is False. Use --device cpu, or --device auto "
                "to fall back automatically."
            )
        return "mps"
    return "mps" if available else "cpu"
