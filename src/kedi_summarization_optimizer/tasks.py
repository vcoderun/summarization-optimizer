"""Autobench task entrypoints; all runtime dependencies are import-addressable."""

from __future__ import annotations

from autobench import Case, RunContext
from pydantic import BaseModel, ConfigDict

from .config import CampaignConfig
from .dataset import load_dataset
from .models import (
    CertificationCaseOutcome,
    HistoryExample,
    OptimizationOutcome,
)
from .optimization import optimize_campaign
from .ports import CheckpointEvaluator, SummarizerInvoker, evaluate, invoke, typed_target


class OptimizationTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: CampaignConfig


class CertificationTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instructions: str
    example: HistoryExample
    invoker_target: str
    evaluator_target: str


def run_optimization(_ctx: RunContext, case: Case) -> OptimizationOutcome:
    task_input = OptimizationTaskInput.model_validate(case.input)
    dataset = load_dataset(task_input.config.dataset_path)
    return optimize_campaign(task_input.config, dataset)


def run_certification(_ctx: RunContext, case: Case) -> CertificationCaseOutcome:
    task_input = CertificationTaskInput.model_validate(case.input)
    invoker = typed_target(task_input.invoker_target, SummarizerInvoker)
    evaluator = typed_target(task_input.evaluator_target, CheckpointEvaluator)
    output = invoke(invoker, task_input.instructions, task_input.example.input)
    result = evaluate(
        evaluator,
        task_input.example.input,
        output,
        task_input.example.expected,
    )
    return CertificationCaseOutcome(
        example_id=task_input.example.id,
        score=result.score,
        hard_pass=result.hard_pass,
        feedback=result.feedback,
        metrics=result.metrics,
    )


__all__ = ("run_certification", "run_optimization")
