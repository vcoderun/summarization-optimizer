"""Deterministic adapters that exercise the real pipeline without model I/O."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic_gepa.values import SerializableValue

from .models import (
    CanonicalAnchor,
    CheckpointEvaluation,
    SummarizationInput,
    SummaryCheckpoint,
)
from .optimization import COMPONENT_NAME

IMPROVED_INSTRUCTIONS = (
    "Preserve continuation-critical facts, canonical anchors, pending work, and uncertainty."
)


def deterministic_invoker(
    instructions: str,
    inputs: SummarizationInput,
) -> SummaryCheckpoint:
    if instructions != IMPROVED_INSTRUCTIONS:
        return SummaryCheckpoint(current_objective="Unknown objective")

    objective = next(
        message.content for message in reversed(inputs.messages) if message.role == "user"
    )
    constraints = _anchor_values(inputs.anchors, "constraint")
    pending = _anchor_values(inputs.anchors, "pending_action")
    decisions = _anchor_values(inputs.anchors, "decision")
    resources = _anchor_values(inputs.anchors, "resource")
    return SummaryCheckpoint(
        current_objective=objective,
        constraints=constraints,
        decisions=decisions,
        resources=resources,
        pending_actions=pending,
        anchors=inputs.anchors,
    )


def exact_evaluator(
    _inputs: SummarizationInput,
    output: SummaryCheckpoint,
    expected: SummaryCheckpoint,
) -> CheckpointEvaluation:
    output_fields = output.model_dump(mode="json")
    expected_fields = expected.model_dump(mode="json")
    matched = sum(output_fields[name] == expected_fields[name] for name in expected_fields)
    score = matched / len(expected_fields)
    hard_pass = output == expected
    return CheckpointEvaluation(
        score=score,
        hard_pass=hard_pass,
        feedback=() if hard_pass else ("Preserve every expected checkpoint field exactly.",),
        metrics={"matched_fields": matched, "total_fields": len(expected_fields)},
    )


def rejecting_evaluator(
    _inputs: SummarizationInput,
    _output: SummaryCheckpoint,
    _expected: SummaryCheckpoint,
) -> CheckpointEvaluation:
    return CheckpointEvaluation(
        score=0.0,
        hard_pass=False,
        feedback=("Rejected by the deterministic certification fixture.",),
    )


def deterministic_proposer(
    candidate: dict[str, str],
    _reflective_dataset: Mapping[str, Sequence[Mapping[str, SerializableValue]]],
    components_to_update: list[str],
) -> dict[str, str]:
    if COMPONENT_NAME not in components_to_update:
        return candidate
    return {**candidate, COMPONENT_NAME: IMPROVED_INSTRUCTIONS}


def _anchor_values(
    anchors: tuple[CanonicalAnchor, ...],
    kind: str,
) -> tuple[str, ...]:
    return tuple(str(anchor.payload["text"]) for anchor in anchors if anchor.kind == kind)


__all__ = (
    "IMPROVED_INSTRUCTIONS",
    "deterministic_invoker",
    "deterministic_proposer",
    "exact_evaluator",
    "rejecting_evaluator",
)
