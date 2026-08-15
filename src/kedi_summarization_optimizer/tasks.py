"""Autobench task entrypoints; all runtime dependencies are import-addressable."""

from __future__ import annotations

import asyncio

from autobench import Case, RunContext
from pydantic import BaseModel, ConfigDict

from .codex_runtime import configure_codex_runtime
from .config import CampaignConfig, CodexModelsSettings
from .dataset import load_dataset
from .models import (
    CertificationCaseOutcome,
    HistoryExample,
    OptimizationOutcome,
)
from .optimization import optimize_campaign
from .ports import CheckpointEvaluator, SummarizerInvoker, aevaluate, ainvoke, typed_target


class OptimizationTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: CampaignConfig


class CertificationTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instructions: str
    example: HistoryExample
    invoker_target: str
    evaluator_target: str
    models: CodexModelsSettings


async def run_optimization(_ctx: RunContext, case: Case) -> OptimizationOutcome:
    task_input = OptimizationTaskInput.model_validate(case.input)
    configure_codex_runtime(task_input.config.models)
    dataset = load_dataset(task_input.config.dataset_path)
    return await asyncio.to_thread(optimize_campaign, task_input.config, dataset)


async def run_certification(_ctx: RunContext, case: Case) -> CertificationCaseOutcome:
    task_input = CertificationTaskInput.model_validate(case.input)
    configure_codex_runtime(task_input.models)
    invoker = typed_target(task_input.invoker_target, SummarizerInvoker)
    evaluator = typed_target(task_input.evaluator_target, CheckpointEvaluator)
    output = await ainvoke(invoker, task_input.instructions, task_input.example.input)
    result = await aevaluate(
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
