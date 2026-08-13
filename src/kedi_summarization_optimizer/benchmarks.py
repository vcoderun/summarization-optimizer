"""Autobench definitions for optimization evidence and independent certification."""

from __future__ import annotations

from autobench import (
    AssetDiscoverySettings,
    Benchmark,
    Case,
    Direction,
    ExactScorer,
    HTTPXInstrumentation,
    ObservationRole,
    PydanticAIInstrumentation,
    PydanticGEPAInstrumentation,
    Semantic,
    Variant,
)

from .config import CampaignConfig
from .models import DatasetBundle, HistoryExample, OptimizationOutcome

TASK_MODULE = "kedi_summarization_optimizer.tasks"


def optimization_benchmark(config: CampaignConfig) -> Benchmark:
    benchmark = (
        Benchmark(f"{config.campaign_id}-optimization")
        .description("Optimize Kedi history summarizer instructions with pydantic-gepa.")
        .dataset(
            [
                Case(
                    id="optimization-campaign",
                    input={"config": config.model_dump(mode="json")},
                    metadata={"phase": "optimization"},
                    tags=["summarization", "optimization", "gepa"],
                )
            ],
            dataset_id=f"{config.campaign_id}-optimizer-control",
            version="1",
        )
        .variants([Variant(id="gepa", label="pydantic-gepa")])
        .task(f"{TASK_MODULE}:run_optimization")
        .instrument(
            PydanticGEPAInstrumentation(
                detail=config.instrumentation.detail,
                assets=AssetDiscoverySettings(
                    discover=True,
                    representations=("definition", "effective"),
                    include=("prompt", "system_prompt"),
                ),
            )
        )
    )
    return _add_model_instrumentation(benchmark, config)


def certification_benchmark(
    config: CampaignConfig,
    dataset: DatasetBundle,
    optimization: OptimizationOutcome,
) -> Benchmark:
    cases = [
        _certification_case(config, optimization.selected_instructions, example)
        for example in dataset.heldout
    ]
    benchmark = (
        Benchmark(f"{config.campaign_id}-certification")
        .description("Independently certify the selected summarizer prompt on held-out histories.")
        .dataset(
            cases,
            dataset_id=f"{config.campaign_id}-heldout",
            version=dataset.version,
            metadata={"fingerprint": dataset.fingerprints()["heldout"]},
        )
        .variants([Variant(id="selected-candidate", label="Selected prompt")])
        .task(f"{TASK_MODULE}:run_certification")
        .scoring(
            [
                ExactScorer(
                    name="hard_pass",
                    semantic_type=Semantic.QUALITY_CORRECTNESS,
                    actual="output.hard_pass",
                    expected="case.expected.hard_pass",
                    direction=Direction.MAXIMIZE,
                    role=ObservationRole.CONSTRAINT,
                )
            ]
        )
    )
    return _add_model_instrumentation(benchmark, config)


def _certification_case(
    config: CampaignConfig,
    instructions: str,
    example: HistoryExample,
) -> Case:
    return Case(
        id=example.id,
        input={
            "instructions": instructions,
            "example": example.model_dump(mode="json"),
            "invoker_target": config.invoker_target,
            "evaluator_target": config.evaluator_target,
        },
        expected={"hard_pass": True},
        metadata={
            "phase": "heldout",
            "scenario_family": example.metadata.scenario_family,
            "generator_version": example.metadata.generator_version,
        },
        tags=["summarization", "heldout", example.metadata.scenario_family],
    )


def _add_model_instrumentation(benchmark: Benchmark, config: CampaignConfig) -> Benchmark:
    if config.instrumentation.include_pydantic_ai:
        benchmark.instrument(PydanticAIInstrumentation())
    if config.instrumentation.include_httpx:
        benchmark.instrument(HTTPXInstrumentation())
    return benchmark


__all__ = ("certification_benchmark", "optimization_benchmark")
