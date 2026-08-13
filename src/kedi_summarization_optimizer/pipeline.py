"""Durable optimize -> replay -> certify -> publish lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path

from autobench import ExecutionCorrelation, record_experiment, replay_experiment

from .benchmarks import certification_benchmark, optimization_benchmark
from .config import CampaignConfig
from .dataset import load_dataset
from .models import (
    CampaignOutcome,
    CertificationCaseOutcome,
    CertificationOutcome,
    OptimizationOutcome,
)


class PipelineError(RuntimeError):
    """Raised when a campaign cannot safely progress to the next phase."""


class SummarizationOptimizationPipeline:
    def __init__(self, config: CampaignConfig, *, config_path: Path | None = None) -> None:
        self.config = config
        self.config_path = config_path

    @classmethod
    def from_file(cls, path: str | Path) -> SummarizationOptimizationPipeline:
        from .config import load_campaign_config

        source = Path(path).expanduser().resolve()
        return cls(load_campaign_config(source), config_path=source)

    def run(self) -> CampaignOutcome:
        config = self.config
        dataset = load_dataset(config.dataset_path)
        self._prepare_output_directory()

        optimization_result = optimization_benchmark(config).run(
            experiment_id=f"{config.campaign_id}-optimization",
            correlation=ExecutionCorrelation(
                group_id=config.campaign_id,
                attempt=1,
                phase="optimization",
                labels={"dataset_version": dataset.version},
            ),
            concurrency_limit=1,
        )
        record_experiment(
            optimization_result,
            config.optimization_record_dir,
            source_files=self._source_files(),
            path_root=self._path_root(),
            durability=config.record_durability,
        )
        replayed_optimization = replay_experiment(config.optimization_record_dir)
        optimization = self._optimization_outcome(replayed_optimization)
        self._write_text("selected_candidate.txt", optimization.selected_instructions)

        certification_result = certification_benchmark(config, dataset, optimization).run(
            experiment_id=f"{config.campaign_id}-certification",
            correlation=ExecutionCorrelation(
                group_id=config.campaign_id,
                attempt=1,
                phase="heldout-certification",
                parent_experiment_id=optimization_result.experiment_id,
                labels={
                    "dataset_version": dataset.version,
                    "candidate": optimization.candidate_fingerprint,
                },
            ),
            concurrency_limit=config.gepa.max_concurrency,
        )
        record_experiment(
            certification_result,
            config.certification_record_dir,
            source_files=self._source_files(),
            path_root=self._path_root(),
            durability=config.record_durability,
        )
        replayed_certification = replay_experiment(config.certification_record_dir)
        certification = self._certification_outcome(replayed_certification)

        accepted_path: str | None = None
        if certification.accepted:
            accepted_path = str(
                self._write_text(
                    "accepted_prompt.txt",
                    optimization.selected_instructions,
                )
            )

        outcome = CampaignOutcome(
            campaign_id=config.campaign_id,
            accepted=certification.accepted,
            selected_candidate_path=str(config.output_dir / "selected_candidate.txt"),
            accepted_prompt_path=accepted_path,
            optimization_record_path=str(config.optimization_record_dir),
            certification_record_path=str(config.certification_record_dir),
            optimization=optimization,
            certification=certification,
        )
        self._write_json("campaign.json", outcome.model_dump(mode="json"))
        return outcome

    def _prepare_output_directory(self) -> None:
        output = self.config.output_dir
        output.mkdir(parents=True, exist_ok=True)
        entries = tuple(output.iterdir())
        resume = self.config.gepa.resume
        if resume == "never":
            if entries:
                raise PipelineError(f"Campaign output directory is not empty: {output}")
            return

        unexpected = tuple(path for path in entries if path != self.config.checkpoint_dir)
        if unexpected:
            names = ", ".join(sorted(path.name for path in unexpected))
            raise PipelineError(
                "A resumed campaign may contain only GEPA checkpoint state; "
                f"found finalized or unrelated output: {names}"
            )
        has_checkpoint = self.config.checkpoint_dir.is_dir() and any(
            self.config.checkpoint_dir.iterdir()
        )
        if resume == "required" and not has_checkpoint:
            raise PipelineError(
                f"resume='required' needs existing checkpoint state in {self.config.checkpoint_dir}"
            )

    def _optimization_outcome(self, experiment: object) -> OptimizationOutcome:
        runs = getattr(experiment, "runs")
        if len(runs) != 1:
            raise PipelineError(f"Expected one optimization run, received {len(runs)}.")
        run = runs[0]
        if run.status.value != "passed":
            message = run.error.message if run.error is not None else "unknown optimization error"
            raise PipelineError(f"Optimization failed: {message}")
        return OptimizationOutcome.model_validate(run.task_result.output)

    def _certification_outcome(self, experiment: object) -> CertificationOutcome:
        runs = getattr(experiment, "runs")
        if not runs:
            raise PipelineError("Certification produced no runs.")
        failed = [run for run in runs if run.status.value != "passed"]
        if failed:
            names = ", ".join(run.case_id for run in failed)
            raise PipelineError(f"Certification runs failed: {names}")
        cases = tuple(
            CertificationCaseOutcome.model_validate(run.task_result.output) for run in runs
        )
        mean_score = sum(case.score for case in cases) / len(cases)
        minimum_score = min(case.score for case in cases)
        policy = self.config.certification
        accepted = (
            mean_score >= policy.minimum_mean_score
            and minimum_score >= policy.minimum_case_score
            and (not policy.require_all_hard_pass or all(case.hard_pass for case in cases))
        )
        return CertificationOutcome(
            accepted=accepted,
            mean_score=mean_score,
            minimum_score=minimum_score,
            cases=cases,
        )

    def _source_files(self) -> list[Path]:
        sources = [self.config.dataset_path]
        if self.config_path is not None:
            sources.append(self.config_path)
        return sources

    def _path_root(self) -> Path:
        roots = [path.parent for path in self._source_files()]
        return Path(os.path.commonpath(roots)) if roots else Path.cwd()

    def _write_text(self, name: str, value: str) -> Path:
        destination = self.config.output_dir / name
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(destination)
        return destination

    def _write_json(self, name: str, value: object) -> Path:
        rendered = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        return self._write_text(name, rendered)


__all__ = ("PipelineError", "SummarizationOptimizationPipeline")
