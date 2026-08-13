from __future__ import annotations

import json
from pathlib import Path

import pytest
from autobench import load_experiment_record, load_run_record, replay_experiment

from kedi_summarization_optimizer import (
    DatasetBundle,
    SummarizationOptimizationPipeline,
    load_campaign_config,
)
from kedi_summarization_optimizer.dataset import to_gepa_split
from kedi_summarization_optimizer.pipeline import PipelineError

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"


def test_smoke_pipeline_records_replays_certifies_and_publishes(tmp_path: Path) -> None:
    raw = json.loads((EXAMPLES / "smoke_campaign.json").read_text(encoding="utf-8"))
    raw["dataset_path"] = str(EXAMPLES / "smoke_dataset.json")
    raw["output_dir"] = str(tmp_path / "campaign")
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    outcome = SummarizationOptimizationPipeline.from_file(config_path).run()

    assert outcome.accepted is True
    assert outcome.certification.mean_score == 1.0
    assert outcome.optimization.best_score == 1.0
    assert outcome.accepted_prompt_path is not None
    assert Path(outcome.accepted_prompt_path).read_text(encoding="utf-8")
    optimization_record_path = Path(outcome.optimization_record_path)
    optimization_record = load_experiment_record(optimization_record_path)
    assert optimization_record.run_count == 1
    optimization_run = load_run_record(
        optimization_record_path / optimization_record.run_paths[0],
        root_dir=optimization_record_path,
    )
    assert "autobench.pydantic_gepa/v1" in optimization_run.extensions
    assert replay_experiment(Path(outcome.certification_record_path)).passed_count == 1


def test_config_and_dataset_are_valid_without_running_models() -> None:
    config = load_campaign_config(EXAMPLES / "smoke_campaign.json")
    dataset = DatasetBundle.model_validate_json(
        (EXAMPLES / "smoke_dataset.json").read_text(encoding="utf-8")
    )

    assert config.gepa.proposer_target is not None
    assert len(dataset.fingerprints()["heldout"]) == 64
    assert to_gepa_split(dataset, seed=0).test == ()


def test_pipeline_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    raw = json.loads((EXAMPLES / "smoke_campaign.json").read_text(encoding="utf-8"))
    raw["dataset_path"] = str(EXAMPLES / "smoke_dataset.json")
    raw["output_dir"] = str(tmp_path / "campaign")
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    pipeline = SummarizationOptimizationPipeline.from_file(config_path)
    pipeline.config.output_dir.mkdir(parents=True)
    (pipeline.config.output_dir / "existing.txt").write_text("evidence", encoding="utf-8")

    with pytest.raises(PipelineError, match="not empty"):
        pipeline.run()


def test_resume_accepts_only_checkpoint_state(tmp_path: Path) -> None:
    raw = json.loads((EXAMPLES / "smoke_campaign.json").read_text(encoding="utf-8"))
    raw["dataset_path"] = str(EXAMPLES / "smoke_dataset.json")
    raw["output_dir"] = str(tmp_path / "resumed-campaign")
    raw["gepa"]["resume"] = "required"
    config_path = tmp_path / "resumed-campaign.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    pipeline = SummarizationOptimizationPipeline.from_file(config_path)

    pipeline.config.checkpoint_dir.mkdir(parents=True)
    (pipeline.config.checkpoint_dir / "state.json").write_text("{}", encoding="utf-8")
    pipeline._prepare_output_directory()

    (pipeline.config.output_dir / "selected_candidate.txt").write_text(
        "already finalized",
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="only GEPA checkpoint state"):
        pipeline._prepare_output_directory()


def test_required_resume_needs_existing_checkpoint(tmp_path: Path) -> None:
    raw = json.loads((EXAMPLES / "smoke_campaign.json").read_text(encoding="utf-8"))
    raw["dataset_path"] = str(EXAMPLES / "smoke_dataset.json")
    raw["output_dir"] = str(tmp_path / "missing-checkpoint")
    raw["gepa"]["resume"] = "required"
    config_path = tmp_path / "missing-checkpoint.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    pipeline = SummarizationOptimizationPipeline.from_file(config_path)

    with pytest.raises(PipelineError, match="needs existing checkpoint state"):
        pipeline._prepare_output_directory()


def test_failed_certification_does_not_publish_accepted_prompt(tmp_path: Path) -> None:
    raw = json.loads((EXAMPLES / "smoke_campaign.json").read_text(encoding="utf-8"))
    raw["dataset_path"] = str(EXAMPLES / "smoke_dataset.json")
    raw["output_dir"] = str(tmp_path / "rejected-campaign")
    raw["evaluator_target"] = "kedi_summarization_optimizer.testing:rejecting_evaluator"
    config_path = tmp_path / "rejected-campaign.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    outcome = SummarizationOptimizationPipeline.from_file(config_path).run()

    assert outcome.accepted is False
    assert outcome.accepted_prompt_path is None
    assert not (Path(raw["output_dir"]) / "accepted_prompt.txt").exists()
    assert (Path(raw["output_dir"]) / "selected_candidate.txt").exists()
