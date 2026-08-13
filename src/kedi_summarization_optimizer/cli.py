"""Command-line entrypoint for validating and running campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_campaign_config
from .dataset import load_dataset
from .pipeline import SummarizationOptimizationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="kedi-summarization-optimize")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("config", type=Path)
    arguments = parser.parse_args()

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
