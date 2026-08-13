"""Dataset loading and conversion to pydantic-gepa examples."""

from __future__ import annotations

from pathlib import Path

from pydantic_gepa import DataSplit, Example

from .models import DatasetBundle, HistoryExample, SummarizationInput, SummaryCheckpoint


def load_dataset(path: str | Path) -> DatasetBundle:
    source = Path(path).expanduser().resolve()
    return DatasetBundle.model_validate_json(source.read_text(encoding="utf-8"))


def to_gepa_split(
    dataset: DatasetBundle,
    *,
    seed: int,
) -> DataSplit[SummarizationInput, SummaryCheckpoint, dict[str, str]]:
    return DataSplit.from_sets(
        train=tuple(_to_example(example) for example in dataset.train),
        validation=tuple(_to_example(example) for example in dataset.validation),
        seed=seed,
    )


def _to_example(
    example: HistoryExample,
) -> Example[SummarizationInput, SummaryCheckpoint, dict[str, str]]:
    return Example(
        id=example.id,
        name=example.id,
        inputs=example.input,
        expected_output=example.expected,
        metadata={
            "scenario_family": example.metadata.scenario_family,
            "generator_version": example.metadata.generator_version,
            **example.metadata.labels,
        },
    )


__all__ = ("load_dataset", "to_gepa_split")
