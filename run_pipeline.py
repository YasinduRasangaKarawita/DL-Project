"""Shared command-line entry point for one-model training and evaluation."""

from __future__ import annotations

import argparse
import json

from src.evaluation.runner import evaluate_checkpoint
from src.models.registry import MODEL_NAMES
from src.training.runner import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate PlantVillage models under the frozen protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train exactly one model; the locked test set is never evaluated.",
    )
    train_parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument("--config", default="configs/config.yaml")
    train_parser.add_argument("--resume", help="Path to a last_resume.pt checkpoint")
    train_parser.add_argument("--run-id", help="Optional explicit run identifier")
    train_parser.add_argument(
        "--artifact-root",
        default=".",
        help="Durable root for models/ and results/ (use mounted Drive in Colab).",
    )
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Override initial epochs for a declared pilot; omit for official runs.",
    )
    train_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit an explicitly provisional pilot from uncommitted code.",
    )

    evaluation_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a selected inference checkpoint on one frozen partition.",
    )
    evaluation_parser.add_argument("--checkpoint", required=True)
    evaluation_parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    evaluation_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluation_parser.add_argument("--output-root", default=".")
    evaluation_parser.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Required safeguard for the team's one-time final locked-test evaluation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        record = train_experiment(
            model_name=args.model,
            seed=args.seed,
            config_path=args.config,
            resume_path=args.resume,
            run_id=args.run_id,
            device_preference=args.device,
            initial_epochs_override=args.epochs,
            allow_dirty=args.allow_dirty,
            artifact_root=args.artifact_root,
        )
        summary = {
            "run_id": record["run_id"],
            "model_name": record["model_name"],
            "seed": record["seed"],
            "epochs_completed": record["epochs_completed"],
            "best_validation_macro_f1": record["best_validation_macro_f1"],
            "artifacts": record["artifacts"],
        }
    else:
        if args.split == "test" and not args.confirm_locked_test:
            parser.error(
                "evaluating the locked test set requires --confirm-locked-test after model selection"
            )
        result = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            split=args.split,
            device_preference=args.device,
            output_root=args.output_root,
        )
        summary = {
            "run_id": result["run_id"],
            "model_name": result["model_name"],
            "split": result["split"],
            "metrics": result["metrics"],
            "result_path": result["result_path"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
