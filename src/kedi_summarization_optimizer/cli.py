"""Command-line entrypoint for validating and running campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_campaign_config
from .dataset import load_dataset
from .generation import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_SEED,
    DEFAULT_TARGET_HISTORY_CHARS,
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
    synthetic_dataset_summary,
    write_synthetic_dataset,
)
from .pipeline import SummarizationOptimizationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="kedi-summarization-optimize")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("config", type=Path)
    generate = subparsers.add_parser("generate-dataset")
    generate.add_argument("output", type=Path)
    generate.add_argument("--version", default=DEFAULT_DATASET_VERSION)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument(
        "--target-history-chars",
        type=int,
        default=DEFAULT_TARGET_HISTORY_CHARS,
    )
    generate.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    if arguments.command == "generate-dataset":
        generation_config = SyntheticDatasetConfig(
            version=arguments.version,
            seed=arguments.seed,
            target_history_chars=arguments.target_history_chars,
        )
        generated = generate_synthetic_dataset(generation_config)
        output = write_synthetic_dataset(
            arguments.output,
            generated,
            overwrite=arguments.force,
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "seed": generation_config.seed,
                    "target_history_chars": generation_config.target_history_chars,
                    **synthetic_dataset_summary(generated),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    config = load_campaign_config(arguments.config)
    dataset = load_dataset(config.dataset_path)
    if arguments.command == "validate":
        print(
            json.dumps(
                {
                    "campaign_id": config.campaign_id,
                    "dataset_version": dataset.version,
                    "fingerprints": dataset.fingerprints(),
                    "split_sizes": {
                        "train": len(dataset.train),
                        "validation": len(dataset.validation),
                        "heldout": len(dataset.heldout),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    outcome = SummarizationOptimizationPipeline(
        config,
        config_path=arguments.config.expanduser().resolve(),
    ).run()
    print(outcome.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
