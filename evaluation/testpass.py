"""Phase 2: the test split, opened once, for both models, and then never again.

Everything about this module is shaped by one fact: it can only be run
honestly a single time. After the test numbers are known, no model may change
and no threshold may be re-picked, because an adjustment made with the test
result in view is fitting to the test set and the reported figure stops
meaning anything.

Code cannot stop a determined person from running this twice. What it can do
is make a second run *visible*:

* :data:`OPENED_MARKER` is written the first time and refused thereafter
  unless the caller passes an explicit reason, which is recorded in the file
  alongside every previous opening. A results table can then be checked
  against the number of times test was read.
* Phase 1's artifact must already exist. The brief's ordering (§4) becomes a
  missing-file error rather than a discipline.
* The standardiser and the threshold come out of the checkpoint and are
  compared against a train-only refit and against the stored value. Neither
  is recomputed here; there is no code path in :mod:`evaluation` that could.

The expected shape of the split is checked too — ~93,000 rows per seed, ~376
positives at 0.41%, and exactly four scenario-3 positives in every seed. Those
are Stage 4's measurements, and a departure from them means the splits file
and the dataset have gone out of step, which would be invisible in an AUPRC.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import MODEL_A_NAME, MODEL_B_NAME, OUTPUT_DIR, SCENARIOS, SEEDS
from evaluation import guards
from evaluation.checkpoints import headline_checkpoints
from evaluation.corpus import Corpus, EvalSplit, EvaluationError
from evaluation.latency import (
    BucketRecall,
    InsiderDetection,
    LatencySummary,
    detections,
    recall_by_position,
    summarise,
)
from evaluation.scoring import Scored, log_scored, score
from modeling.metrics import aggregate

logger = logging.getLogger(__name__)

#: The audit trail. Every opening of the test split appends to this file.
OPENED_MARKER = OUTPUT_DIR / "TEST_SPLIT_OPENED.json"

MODELS = (MODEL_A_NAME, MODEL_B_NAME)

#: Stage 4's measured test-split shape, used only as a tripwire. A run that
#: departs from these has a splits/dataset mismatch, not a modelling result.
EXPECTED_ROWS_PER_SEED = 93_000
EXPECTED_POSITIVES_PER_SEED = 376
EXPECTED_SCENARIO_3_POSITIVES = 4
#: How far the measured shape may drift before the tripwire fires. Generous,
#: because it is checking for a structural break, not for noise.
_SHAPE_TOLERANCE = 0.10


class PhaseOrderError(RuntimeError):
    """Phase 2 was reached before Phase 1 finished. See brief §4."""


@dataclass(slots=True)
class TestResult:
    """The final numbers, plus every guard that says they can be believed."""

    seeds: list[int]
    sample: bool
    device: str
    scored: list[Scored] = field(default_factory=list)
    detections: list[InsiderDetection] = field(default_factory=list)
    latency: list[LatencySummary] = field(default_factory=list)
    buckets: list[BucketRecall] = field(default_factory=list)
    guards: list[guards.GuardResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.guards if g.fatal)

    def for_model(self, model: str) -> list[Scored]:
        return [s for s in self.scored if s.model_name == model]

    def headline(self, model: str) -> dict[str, Any]:
        """AUPRC, AUROC and P@k as mean +/- sd across seeds, with the baseline.

        AUPRC may be quoted this way — Stage 5 measured it stable across seeds.
        Recall at threshold may not, which is why it is absent from this
        method and pooled as counts in :meth:`per_scenario` instead.
        """
        items = self.for_model(model)
        if not items:
            return {}
        evaluations = [item.evaluation for item in items]
        return {
            "model": model,
            "context_days": items[0].checkpoint.context_days,
            "n_parameters": items[0].checkpoint.n_parameters,
            "seeds": [item.seed for item in items],
            "auprc": aggregate([e.auprc for e in evaluations]),
            "random_baseline": aggregate([e.random_baseline for e in evaluations]),
            "lift_over_random": aggregate([e.lift for e in evaluations]),
            "auroc": aggregate([e.auroc for e in evaluations if e.auroc is not None]),
            "precision_at_10": aggregate(
                [e.precision_at_k[10] for e in evaluations]
            ),
            "precision_at_50": aggregate(
                [e.precision_at_k[50] for e in evaluations]
            ),
            "rows_flagged": aggregate(
                [float(int(item.flagged.sum())) for item in items]
            ),
            "per_seed_auprc": {
                str(item.seed): item.evaluation.auprc for item in items
            },
        }

    def per_scenario(self, model: str) -> dict[str, Any]:
        """Pooled hits and attack days per scenario, summed across seeds.

        Counts, not a mean of rates. Scenario 3 contributes exactly four
        positive rows per split; "2 of 4 attack days" is honest and "50%" is a
        percentage doing an impression of a measurement.
        """
        out: dict[str, Any] = {}
        for scenario in SCENARIOS:
            hits = total = 0
            for item in self.for_model(model):
                for entry in item.evaluation.per_scenario:
                    if entry.scenario == scenario:
                        hits += entry.hits
                        total += entry.total
            out[str(scenario)] = {
                "hits_summed": hits,
                "attack_days_summed": total,
                "pooled_recall": hits / total if total else None,
            }
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": 2,
            "split": "test",
            "sample": self.sample,
            "device": self.device,
            "seeds": self.seeds,
            "headline": {model: self.headline(model) for model in MODELS},
            "per_scenario_pooled": {
                model: self.per_scenario(model) for model in MODELS
            },
            "detection_latency": {
                "summaries": [s.as_dict() for s in self.latency],
                "position_buckets": [b.as_dict() for b in self.buckets],
                "per_insider": [d.as_dict() for d in self.detections],
            },
            "per_seed": [item.as_dict() for item in self.scored],
            "guards": [g.as_dict() for g in self.guards],
            "all_guards_passed": self.passed,
        }


def require_phase1(*, sample: bool = False, directory: Path = OUTPUT_DIR) -> Path:
    """Refuse to open test until Phase 1 has written its artifact.

    §4: once the test split is opened, no model may change. Every remaining
    question therefore gets answered on validation *first*. This turns that
    rule into a precondition instead of a resolution.
    """
    suffix = ".sample" if sample else ""
    artifact = directory / f"phase1_validation{suffix}.json"
    if not artifact.is_file():
        raise PhaseOrderError(
            f"Phase 1 has not been run: {artifact} does not exist.\n"
            f"Brief §4 — once the test split is opened no model may change, "
            f"so every validation analysis must finish first.\n"
            f"Run:  python -m evaluation.run_phase1"
            f"{' --sample' if sample else ''}"
        )
    return artifact


def record_opening(
    *, reason: str, seeds: Sequence[int], sample: bool, force: bool
) -> dict[str, Any]:
    """Append to the audit trail, refusing a second opening without a reason.

    The refusal is not security. It is a note to the person doing it, at the
    moment they are doing it, that the number they are about to produce is no
    longer a single-pass test result and the report has to say so.
    """
    marker = OPENED_MARKER
    history: list[dict[str, Any]] = []
    if marker.is_file():
        previous = json.loads(marker.read_text())
        history = previous.get("openings", [])

    real_openings = [entry for entry in history if not entry.get("sample")]
    if real_openings and not sample and not force:
        when = real_openings[-1]["when"]
        raise EvaluationError(
            f"the test split was already opened at {when} "
            f"({len(real_openings)} time(s) so far).\n"
            f"A second pass is not a second experiment: any change made "
            f"between the two was informed by test numbers.\n"
            f"If you must re-run — a bug in the evaluation code, not a "
            f"disappointing result — pass --force --reason '...' and the "
            f"reason is recorded in {marker.name} for the report to disclose."
        )

    entry = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": list(seeds),
        "sample": sample,
        "forced": force,
        "reason": reason,
    }
    history.append(entry)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"openings": history}, indent=2) + "\n")
    return entry


def run(
    *,
    seeds: Sequence[int] = SEEDS,
    sample: bool = False,
    device: str = "cpu",
    corpus: Corpus | None = None,
) -> TestResult:
    """One pass. Both models. Three seeds each. Then stop.

    Per seed: load both checkpoints, prove they carry the same standardiser as
    a train-only refit, build the test rows under it, apply each checkpoint's
    own stored threshold, and score. Nothing is fitted, chosen or swept.
    """
    corpus = corpus or Corpus.load(sample=sample)
    result = TestResult(seeds=list(seeds), sample=sample, device=device)

    result.guards.append(guards.split_disjointness(corpus, seeds))

    for seed in seeds:
        logger.info("=" * 72)
        logger.info("TEST | seed %d", seed)
        logger.info("=" * 72)

        checkpoints = headline_checkpoints(seed, sample=sample)
        for checkpoint in checkpoints.values():
            logger.info("  %s", checkpoint.describe())
            logger.info(
                "    standardiser from checkpoint: mean[0..2]=%s std[0..2]=%s",
                " ".join(f"{v:+.6f}" for v in checkpoint.standardiser.mean[:3]),
                " ".join(f"{v:.6f}" for v in checkpoint.standardiser.std[:3]),
            )

        result.guards.append(
            guards.standardiser_from_checkpoint(corpus, seed, checkpoints.values())
        )

        splits: list[EvalSplit] = []
        for checkpoint in checkpoints.values():
            split = corpus.build("test", seed, checkpoint.standardiser)
            splits.append(split)
            logger.info("  %s", split.describe())
            item = score(checkpoint, split, device=device)
            log_scored(item)
            result.scored.append(item)
            result.detections.extend(detections(item))

        result.guards.append(guards.identical_rows(seed, splits))
        result.guards.append(_shape_guard(seed, splits[0]))

    result.guards.extend(
        [
            guards.threshold_from_checkpoint(result.scored),
            guards.finite_predictions(result.scored),
            guards.beats_random(result.scored),
            guards.not_suspiciously_good(result.scored),
        ]
    )

    for model in MODELS:
        if not result.for_model(model):
            continue
        result.latency.append(summarise(result.detections, model))
        for scenario in SCENARIOS:
            summary = summarise(result.detections, model, scenario)
            if summary.n_insiders:
                result.latency.append(summary)
    result.buckets.extend(recall_by_position(result.scored))
    for scenario in SCENARIOS:
        result.buckets.extend(recall_by_position(result.scored, scenario))

    log_headline(result)
    guards.report(result.guards)
    return result


def _shape_guard(seed: int, split: EvalSplit) -> guards.GuardResult:
    """The test split looks the way Stage 4 measured it.

    Non-fatal: on ``--sample`` the shape is deliberately different, and even
    on the full release a small drift is information rather than a stop. A
    large one means the splits file and the dataset disagree about who exists,
    which no metric would reveal.
    """
    scenario_3 = int(((split.labels == 1) & (split.scenarios == 3)).sum())
    detail = (
        f"{split.n_rows:,} rows (expected ~{EXPECTED_ROWS_PER_SEED:,}), "
        f"{split.n_positives} positives (expected ~{EXPECTED_POSITIVES_PER_SEED}, "
        f"{split.random_baseline:.4%}), scenario-3 positives {scenario_3} "
        f"(expected exactly {EXPECTED_SCENARIO_3_POSITIVES})"
    )
    name = f"seed {seed}: test split matches Stage 4's measured shape"
    off = (
        abs(split.n_rows - EXPECTED_ROWS_PER_SEED) > _SHAPE_TOLERANCE * EXPECTED_ROWS_PER_SEED
        or abs(split.n_positives - EXPECTED_POSITIVES_PER_SEED)
        > _SHAPE_TOLERANCE * EXPECTED_POSITIVES_PER_SEED
        or scenario_3 != EXPECTED_SCENARIO_3_POSITIVES
    )
    return guards.GuardResult(name, not off, detail, fatal=False)


def log_headline(result: TestResult) -> None:
    """The table §12 opens with, printed at the end of the run."""
    logger.info("=" * 72)
    logger.info("TEST RESULTS — %s", "SAMPLE (meaningless)" if result.sample else "full r4.2")
    logger.info("=" * 72)
    for model in MODELS:
        head = result.headline(model)
        if not head:
            continue
        logger.info(
            "%-8s (%2d-day context, %5s params)  AUPRC %.4f +/- %.4f   "
            "random %.5f  =>  %.1fx",
            model, head["context_days"], f"{head['n_parameters']:,}",
            head["auprc"]["mean"], head["auprc"]["std"],
            head["random_baseline"]["mean"], head["lift_over_random"]["mean"],
        )
        logger.info(
            "%-8s  AUROC %.4f +/- %.4f   P@10 %.3f +/- %.3f   P@50 %.3f +/- %.3f",
            "", head["auroc"]["mean"], head["auroc"]["std"],
            head["precision_at_10"]["mean"], head["precision_at_10"]["std"],
            head["precision_at_50"]["mean"], head["precision_at_50"]["std"],
        )
        for scenario, entry in sorted(result.per_scenario(model).items()):
            pooled = entry["pooled_recall"]
            logger.info(
                "%-8s  scenario %s: %d of %d attack days across all seeds%s",
                "", scenario, entry["hits_summed"], entry["attack_days_summed"],
                f"  ({pooled:.1%})" if pooled is not None else "",
            )


def write(result: TestResult, *, directory: Path = OUTPUT_DIR) -> Path:
    """Write the test artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".sample" if result.sample else ""
    path = directory / f"phase2_test{suffix}.json"
    path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n")
    logger.info("phase 2 -> %s", path)
    return path
