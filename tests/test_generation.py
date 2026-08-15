from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kedi_summarization_optimizer.cli import main
from kedi_summarization_optimizer.dataset import load_dataset
from kedi_summarization_optimizer.evaluation import deterministic_evaluator
from kedi_summarization_optimizer.generation import (
    GENERATOR_VERSION,
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
    synthetic_dataset_summary,
    write_synthetic_dataset,
)

TARGET_HISTORY_CHARS = 4_000
PROJECT_ROOT = Path(__file__).parents[1]
SYNTHETIC_V1_FINGERPRINTS = {
    "train": "baea88d0903de585fd24a2b5316afcdb0cf426c9dc58342712510b29dd8e3a8b",
    "validation": "691fdef26936bcf3a81d4df97d6062e85a0452491ec33dbc1f4f0bab329ba6ba",
    "heldout": "acd3e03328c4ec78251bc39ec1075ecc520541e1ea3736e2ded59a09c3d3c40d",
}
REQUIRED_FAMILIES = {
    "approval-lifecycle",
    "artifact-lifecycle",
    "false-completion",
    "interrupted-stream",
    "latest-wins",
    "parallel-tools",
    "resolved-failure",
    "scoped-exception",
    "secret-artifact",
    "stale-plan-resource",
    "subagent-retry",
    "workflow-cancellation",
}


def _config(*, seed: int = 42) -> SyntheticDatasetConfig:
    return SyntheticDatasetConfig(
        version="test-synthetic-v1",
        seed=seed,
        target_history_chars=TARGET_HISTORY_CHARS,
    )


def test_generation_is_deterministic_isolated_and_high_context() -> None:
    first = generate_synthetic_dataset(_config())
    second = generate_synthetic_dataset(_config())

    assert first == second
    assert first.fingerprints() == second.fingerprints()
    assert (len(first.train), len(first.validation), len(first.heldout)) == (12, 6, 6)

    split_families = [
        {example.metadata.scenario_family for example in split}
        for split in (first.train, first.validation, first.heldout)
    ]
    assert set.union(*split_families) == REQUIRED_FAMILIES
    assert not split_families[0] & split_families[1]
    assert not split_families[0] & split_families[2]
    assert not split_families[1] & split_families[2]

    for example in (*first.train, *first.validation, *first.heldout):
        assert example.metadata.generator_version == GENERATOR_VERSION
        assert sum(len(message.content) for message in example.input.messages) >= (
            TARGET_HISTORY_CHARS
        )
        evaluation = deterministic_evaluator(
            example.input,
            example.expected,
            example.expected,
        )
        assert evaluation.hard_pass, (example.id, evaluation.feedback)
        assert evaluation.score == 1.0


def test_seed_changes_only_generated_history_and_fingerprints() -> None:
    first = generate_synthetic_dataset(_config(seed=1))
    second = generate_synthetic_dataset(_config(seed=2))

    assert first.fingerprints() != second.fingerprints()
    first_expected = [item.expected for item in (*first.train, *first.validation, *first.heldout)]
    second_expected = [
        item.expected for item in (*second.train, *second.validation, *second.heldout)
    ]
    assert first_expected == second_expected


def test_synthetic_canaries_are_source_only() -> None:
    dataset = generate_synthetic_dataset(_config())
    secret_examples = tuple(
        example
        for example in dataset.heldout
        if example.metadata.scenario_family == "secret-artifact"
    )

    assert len(secret_examples) == 2
    for example in secret_examples:
        source = "\n".join(message.content for message in example.input.messages)
        expected = example.expected.model_dump_json()
        assert example.input.evaluation.sensitive_values
        for canary in example.input.evaluation.sensitive_values:
            assert canary.startswith("SYNTHETIC_CANARY_")
            assert canary in source
            assert canary not in expected


def test_lifecycle_families_cover_pending_and_non_live_states() -> None:
    dataset = generate_synthetic_dataset(_config())
    train_by_family: dict[str, list[str]] = {}
    for example in dataset.train:
        train_by_family.setdefault(example.metadata.scenario_family, []).append(
            "\n".join(message.content for message in example.input.messages)
        )
    validation_by_family: dict[str, list[str]] = {}
    for example in dataset.validation:
        validation_by_family.setdefault(example.metadata.scenario_family, []).append(
            "\n".join(message.content for message in example.input.messages)
        )

    approval_text = "\n".join(train_by_family["approval-lifecycle"])
    assert "decision=edit" in approval_text
    assert "decision=deny" in approval_text
    assert "decision=pending" in approval_text

    artifact_text = "\n".join(train_by_family["artifact-lifecycle"])
    for state in ("expired", "released", "missing", "oversized"):
        assert state in artifact_text

    interrupted_text = "\n".join(validation_by_family["interrupted-stream"])
    assert "Reasoning block" in interrupted_text
    assert "stream interrupted by cancellation" in interrupted_text


def test_atomic_writer_round_trips_and_requires_explicit_overwrite(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(_config())
    destination = tmp_path / "datasets" / "synthetic.json"

    assert write_synthetic_dataset(destination, dataset) == destination.resolve()
    assert load_dataset(destination) == dataset
    with pytest.raises(FileExistsError, match="Dataset already exists"):
        write_synthetic_dataset(destination, dataset)
    assert write_synthetic_dataset(destination, dataset, overwrite=True) == destination.resolve()


def test_cli_generates_dataset_and_publication_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "synthetic.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kedi-summarization-optimize",
            "generate-dataset",
            str(destination),
            "--version",
            "cli-synthetic-v1",
            "--seed",
            "19",
            "--target-history-chars",
            str(TARGET_HISTORY_CHARS),
        ],
    )

    main()

    summary = json.loads(capsys.readouterr().out)
    dataset = load_dataset(destination)
    assert summary["output"] == str(destination.resolve())
    assert summary["seed"] == 19
    assert summary["fingerprints"] == dataset.fingerprints()
    assert summary["splits"]["heldout"]["examples"] == 6
    assert synthetic_dataset_summary(dataset)["generator_version"] == GENERATOR_VERSION


def test_checked_in_synthetic_v1_matches_the_default_generator() -> None:
    checked_in = load_dataset(PROJECT_ROOT / "datasets" / "synthetic_v1.json")

    assert checked_in == generate_synthetic_dataset()
    assert checked_in.fingerprints() == SYNTHETIC_V1_FINGERPRINTS
