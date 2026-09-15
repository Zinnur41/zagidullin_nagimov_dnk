"""Генерация промоторов, оценка модели и отчёт в ClearML."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

import joblib
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from promoter_gen.generator import GenerationConfig, PromoterGenerator, STRENGTHS
from promoter_gen.io import dataset_summary, write_csv, write_fasta, write_json
from promoter_gen.model import promoter_probabilities, train_and_evaluate
from promoter_gen.plots import all_figures
from promoter_gen.tracking import ClearMLTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-promoters", type=int, default=1_200)
    parser.add_argument("--n-background", type=int, default=1_200)
    parser.add_argument("--n-generated", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--project-name", default="DNA Promoter Generation")
    parser.add_argument(
        "--task-name",
        default=None,
        help="ClearML task name; generated from the seed when omitted",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--no-clearml", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Проверить размеры выборок и число кандидатов."""
    for name in ("n_promoters", "n_background", "n_generated"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    if min(args.n_promoters, args.n_background) < 10:
        raise ValueError("at least 10 examples of each class are required")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Выполнить эксперимент и сохранить результаты."""
    args = build_parser().parse_args(argv)
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_name = args.task_name or f"sigma70-generator-seed-{args.seed}"

    tracker = ClearMLTracker.create(
        enabled=not args.no_clearml,
        offline=args.offline,
        project_name=args.project_name,
        task_name=task_name,
    )

    config = GenerationConfig()
    if tracker:
        tracker.connect_parameters(asdict(config), "generation")
        tracker.connect_parameters(
            {
                "n_promoters": args.n_promoters,
                "n_background": args.n_background,
                "n_generated_per_strength": args.n_generated,
                "seed": args.seed,
                "classifier": "char TF-IDF (3-6-mers) + logistic regression",
                "test_size": 0.20,
                "cross_validation_folds": 5,
            },
            "experiment",
        )

    print("1/5 Generating the reference dataset...")
    training_generator = PromoterGenerator(config=config, seed=args.seed)
    records = training_generator.generate_dataset(
        n_promoters=args.n_promoters,
        n_background=args.n_background,
    )
    dataset_csv = write_csv(records, output_dir / "reference_dataset.csv")
    dataset_fasta = write_fasta(records, output_dir / "reference_dataset.fasta")

    print("2/5 Training and evaluating the k-mer classifier...")
    result = train_and_evaluate(records, seed=args.seed)
    model_path = output_dir / "promoter_classifier.joblib"
    joblib.dump(result.model, model_path)

    print("3/5 Generating unseen promoter candidates...")
    candidate_generator = PromoterGenerator(config=config, seed=args.seed + 10_000)
    generated_records = candidate_generator.generate_promoters(
        args.n_generated * len(STRENGTHS),
        strengths=STRENGTHS,
        identifier_prefix="C",
    )
    generated_probability = promoter_probabilities(
        result.model,
        [record.sequence for record in generated_records],
    )
    candidates_fasta = write_fasta(
        generated_records,
        output_dir / "generated_candidates.fasta",
    )
    candidates_csv = write_csv(
        generated_records,
        output_dir / "generated_candidates.csv",
    )

    cv_summary = {
        metric: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": values,
        }
        for metric, values in result.cross_validation.items()
    }
    generated_by_strength = {}
    for strength in STRENGTHS:
        selection = [
            float(probability)
            for record, probability in zip(generated_records, generated_probability)
            if record.strength == strength
        ]
        generated_by_strength[strength] = {
            "count": len(selection),
            "mean_promoter_probability": float(np.mean(selection)),
            "std_promoter_probability": float(np.std(selection)),
        }
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "dataset": dataset_summary(records),
        "generated_candidates": dataset_summary(generated_records),
        "holdout_metrics": result.metrics,
        "cross_validation": cv_summary,
        "generated_quality_by_strength": generated_by_strength,
    }
    summary_path = write_json(summary, output_dir / "metrics.json")

    print("4/5 Building interactive Plotly figures...")
    figures = all_figures(
        records=records,
        generated_records=generated_records,
        generated_probabilities=generated_probability,
        config=config,
        false_positive_rate=result.false_positive_rate,
        true_positive_rate=result.true_positive_rate,
        roc_auc=result.metrics["roc_auc"],
        confusion=result.confusion,
        cross_validation=result.cross_validation,
    )

    if tracker:
        print("5/5 Reporting interactive figures and artifacts to ClearML...")
        for metric, value in result.metrics.items():
            tracker.report_scalar("Hold-out metrics", metric, value, iteration=0)
        for metric, values in result.cross_validation.items():
            for fold, value in enumerate(values, start=1):
                tracker.report_scalar(
                    "Cross-validation",
                    metric,
                    value,
                    iteration=fold,
                )
        for title, figure in figures.items():
            tracker.report_plotly(title=title, figure=figure)
        for name, path in {
            "reference_dataset_csv": dataset_csv,
            "reference_dataset_fasta": dataset_fasta,
            "generated_candidates_csv": candidates_csv,
            "generated_candidates_fasta": candidates_fasta,
            "trained_classifier": model_path,
            "experiment_metrics": summary_path,
        }.items():
            tracker.upload_artifact(name, path)
        if args.offline:
            print(f"ClearML offline session: {tracker.offline_folder()}")
        else:
            print(f"ClearML task: {tracker.web_url()}")
        tracker.close()
    else:
        print("5/5 ClearML disabled; local artifacts are ready.")

    print(json.dumps(summary["holdout_metrics"], indent=2, sort_keys=True))
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())