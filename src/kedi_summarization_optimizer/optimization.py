"""pydantic-gepa optimization assembly for one campaign."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from pydantic_gepa import (
    CandidateContext,
    Component,
    DerivedValueInjection,
    MetricResult,
    Optimization,
    ScoreObjective,
)
from pydantic_gepa.configuration import (
    BudgetConfig,
    EvaluationSetConfig,
    GEPAConfig,
    ReflectionConfig,
    RunConfig,
)
from pydantic_gepa.reflection import ReflectionFunction

from .config import CampaignConfig
from .dataset import to_gepa_split
from .models import (
    DatasetBundle,
    OptimizationOutcome,
    SummarizationInput,
    SummaryCheckpoint,
)
from .ports import (
    CandidateProposer,
    CheckpointEvaluator,
    SummarizerInvoker,
    aevaluate,
    ainvoke,
    resolve_target,
    typed_target,
)

COMPONENT_NAME = "summarizer_instructions"
OBJECTIVE_NAME = "checkpoint_quality"


def optimize_campaign(
    config: CampaignConfig,
    dataset: DatasetBundle,
) -> OptimizationOutcome:
    active_instructions = CandidateContext[str]("kedi.summarizer.instructions")
    component = Component(
        name=COMPONENT_NAME,
        initial_text=config.seed_instructions,
        kind="system_prompt",
        semantic_type="kedi.history.summarizer.instructions",
        source="kedi_summarization_optimizer",
        path="summarizer_instructions",
    )
    invoker = typed_target(config.invoker_target, SummarizerInvoker)
    evaluator = typed_target(config.evaluator_target, CheckpointEvaluator)
    data = to_gepa_split(dataset, seed=config.gepa.seed)

    async def run_target(inputs: SummarizationInput) -> SummaryCheckpoint:
        return await ainvoke(invoker, active_instructions.require(), inputs)

    async def score_target(context: object) -> Mapping[str, MetricResult]:
        inputs = cast("SummarizationInput", getattr(context, "inputs"))
        output = cast("SummaryCheckpoint", getattr(context, "output"))
        expected = cast("SummaryCheckpoint | None", getattr(context, "expected_output"))
        if expected is None:
            raise ValueError("Every optimization example must define expected_output.")
        result = await aevaluate(evaluator, inputs, output, expected)
        feedback = "\n".join(result.feedback) or None
        return {
            OBJECTIVE_NAME: MetricResult(
                score=result.score,
                role="objective",
                feedback=feedback,
                side_info={"hard_pass": result.hard_pass, **result.metrics},
                semantic_type="quality.correctness",
                direction="maximize",
            ),
            "hard_pass": MetricResult(
                score=float(result.hard_pass),
                role="constraint",
                feedback=feedback,
                semantic_type="quality.correctness",
                direction="maximize",
            ),
        }

    optimization = Optimization.from_examples(
        data=data,
        task=run_target,
        score=score_target,
        score_key=OBJECTIVE_NAME,
        objective=ScoreObjective(score_key=OBJECTIVE_NAME, direction="maximize"),
        components=(component,),
        injections=(
            DerivedValueInjection(
                component=COMPONENT_NAME,
                context=active_instructions,
                required_components=(COMPONENT_NAME,),
                derive_value=lambda candidate: component.decode(candidate[COMPONENT_NAME]),
            ),
        ),
        max_concurrency=config.gepa.max_concurrency,
        dataset_name=f"{config.campaign_id}-summarization",
    )
    result = optimization.optimize(config=_gepa_config(config, dataset))
    selected = component.decode(result.best_candidate.values[COMPONENT_NAME])
    return OptimizationOutcome(
        selected_instructions=selected,
        candidate_id=result.best_candidate.id or result.best_candidate.fingerprint(),
        candidate_fingerprint=result.best_candidate.fingerprint(),
        best_score=result.best_score,
        validation_score=result.scores.validation,
        metric_calls=result.total_metric_calls,
        stop_reason=result.stop_reason,
        dataset_fingerprints=dataset.fingerprints(),
    )


def _gepa_config(config: CampaignConfig, dataset: DatasetBundle) -> GEPAConfig:
    proposer = (
        None
        if config.gepa.proposer_target is None
        else typed_target(config.gepa.proposer_target, CandidateProposer)
    )
    reflection_model: str | ReflectionFunction | None = config.gepa.reflection_model
    if config.gepa.reflection_target is not None:
        reflection_model = cast("ReflectionFunction", resolve_target(config.gepa.reflection_target))

    return GEPAConfig(
        reflection=ReflectionConfig(
            model=reflection_model,
            proposer=proposer,
            minibatch_size=config.gepa.reflection_minibatch_size,
        ),
        budget=BudgetConfig(max_metric_calls=config.gepa.max_metric_calls),
        run=RunConfig(
            id=config.campaign_id,
            directory=config.checkpoint_dir,
            resume=config.gepa.resume,
            fresh=config.gepa.fresh,
            checkpoint_interval=config.gepa.checkpoint_interval,
            compatibility={
                "dataset_version": dataset.version,
                **{
                    f"dataset_{name}": fingerprint
                    for name, fingerprint in dataset.fingerprints().items()
                    if name != "heldout"
                },
                "invoker_target": config.invoker_target,
                "evaluator_target": config.evaluator_target,
                "models": config.models.model_dump_json(),
                "reflection_strategy": json.dumps(
                    {
                        "model": config.gepa.reflection_model,
                        "reflection_target": config.gepa.reflection_target,
                        "proposer_target": config.gepa.proposer_target,
                        "minibatch_size": config.gepa.reflection_minibatch_size,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
            seed=config.gepa.seed,
            cache_evaluations=config.gepa.cache_evaluations,
        ),
        evaluation_sets=EvaluationSetConfig(allow_same_train_validation=False),
    )


__all__ = ("COMPONENT_NAME", "OBJECTIVE_NAME", "optimize_campaign")
