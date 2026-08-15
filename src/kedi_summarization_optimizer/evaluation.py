"""Deterministic quality and safety evaluation for summary checkpoints."""

from __future__ import annotations

import json
import re

from .models import CheckpointEvaluation, SummarizationInput, SummaryCheckpoint

_SEMANTIC_FIELDS = (
    "operating_rules",
    "project_context",
    "constraints",
    "decisions",
)
_ACTIVE_STATE_FIELDS = (
    "current_objective",
    "current_execution_state",
    "unresolved_problems",
    "pending_actions",
)
_EPISODIC_FIELDS = (
    "recent_progress",
    "completed_actions",
    "lifecycle_outcomes",
    "resources",
    "uncertainties",
)


def deterministic_evaluator(
    inputs: SummarizationInput,
    output: SummaryCheckpoint,
    expected: SummaryCheckpoint,
) -> CheckpointEvaluation:
    """Score reconstructive fidelity while enforcing non-negotiable constraints."""

    serialized = output.model_dump_json()
    searchable = _searchable_text(output)
    normalized_searchable = _normalize(searchable)
    contract = inputs.evaluation

    output_anchor_ids = tuple(anchor.id for anchor in output.anchors)
    input_anchors = {anchor.id: anchor for anchor in inputs.anchors}
    expected_anchors = {anchor.id: anchor for anchor in expected.anchors}
    output_anchors = {anchor.id: anchor for anchor in output.anchors}
    output_artifact_ids = output.artifact_ids
    failures: list[str] = []

    if len(serialized) > inputs.max_output_chars:
        failures.append(
            f"Checkpoint exceeds max_output_chars ({len(serialized)} > {inputs.max_output_chars})."
        )

    leaked_forbidden = tuple(
        phrase
        for phrase in contract.forbidden_phrases
        if _normalize(phrase) in normalized_searchable
    )
    if leaked_forbidden:
        failures.append("Checkpoint retains superseded or explicitly forbidden information.")

    leaked_sensitive = tuple(value for value in contract.sensitive_values if value in searchable)
    if leaked_sensitive:
        failures.append("Checkpoint exposes sensitive values from the source history.")

    missing_anchor_ids = tuple(
        anchor_id for anchor_id in expected_anchors if anchor_id not in output_anchors
    )
    if missing_anchor_ids:
        failures.append("Checkpoint omits required canonical anchors.")

    invented_anchor_ids = tuple(
        anchor_id for anchor_id in output_anchor_ids if anchor_id not in input_anchors
    )
    if invented_anchor_ids:
        failures.append("Checkpoint invents canonical anchors not present in the source history.")

    mismatched_anchor_ids = tuple(
        anchor_id
        for anchor_id, anchor in output_anchors.items()
        if anchor_id in expected_anchors and anchor != expected_anchors[anchor_id]
    )
    if mismatched_anchor_ids:
        failures.append("Checkpoint changes canonical anchor payloads.")

    missing_artifact_ids = tuple(
        artifact_id
        for artifact_id in expected.artifact_ids
        if artifact_id not in output_artifact_ids
    )
    if missing_artifact_ids:
        failures.append("Checkpoint omits required artifact references.")

    invented_artifact_ids = tuple(
        artifact_id
        for artifact_id in output_artifact_ids
        if artifact_id not in expected.artifact_ids
    )
    if invented_artifact_ids:
        failures.append("Checkpoint invents artifact references not present in expected state.")

    if _has_duplicate_entries(output):
        failures.append("Checkpoint contains duplicate entries within a section.")

    semantic_score = _group_score(output, expected, _SEMANTIC_FIELDS)
    active_state_score = _group_score(output, expected, _ACTIVE_STATE_FIELDS)
    episodic_score = _group_score(output, expected, _EPISODIC_FIELDS)
    reference_score = _reference_score(output, expected)
    score = (
        semantic_score * 0.55
        + active_state_score * 0.25
        + episodic_score * 0.15
        + reference_score * 0.05
    )

    metrics: dict[str, float | int | bool] = {
        "semantic_score": semantic_score,
        "active_state_score": active_state_score,
        "episodic_score": episodic_score,
        "reference_score": reference_score,
        "output_chars": len(serialized),
        "forbidden_leaks": len(leaked_forbidden),
        "sensitive_leaks": len(leaked_sensitive),
        "missing_anchors": len(missing_anchor_ids),
        "invented_anchors": len(invented_anchor_ids),
        "mismatched_anchors": len(mismatched_anchor_ids),
        "missing_artifacts": len(missing_artifact_ids),
        "invented_artifacts": len(invented_artifact_ids),
    }
    return CheckpointEvaluation(
        score=score,
        hard_pass=not failures,
        feedback=tuple(failures),
        metrics=metrics,
    )


def _group_score(
    output: SummaryCheckpoint,
    expected: SummaryCheckpoint,
    fields: tuple[str, ...],
) -> float:
    scores = tuple(_field_score(getattr(output, name), getattr(expected, name)) for name in fields)
    return sum(scores) / len(scores)


def _field_score(output: str | tuple[str, ...], expected: str | tuple[str, ...]) -> float:
    if isinstance(output, str) and isinstance(expected, str):
        return float(_normalize(output) == _normalize(expected))
    if isinstance(output, tuple) and isinstance(expected, tuple):
        output_values = {_normalize(value) for value in output}
        expected_values = {_normalize(value) for value in expected}
        if not output_values and not expected_values:
            return 1.0
        if not output_values or not expected_values:
            return 0.0
        overlap = len(output_values & expected_values)
        if not overlap:
            return 0.0
        precision = overlap / len(output_values)
        recall = overlap / len(expected_values)
        return 2 * precision * recall / (precision + recall)
    raise TypeError("Checkpoint field types must match.")


def _reference_score(output: SummaryCheckpoint, expected: SummaryCheckpoint) -> float:
    anchor_score = _field_score(
        tuple(anchor.id for anchor in output.anchors),
        tuple(anchor.id for anchor in expected.anchors),
    )
    artifact_score = _field_score(output.artifact_ids, expected.artifact_ids)
    return (anchor_score + artifact_score) / 2


def _searchable_text(checkpoint: SummaryCheckpoint) -> str:
    return json.dumps(checkpoint.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _has_duplicate_entries(checkpoint: SummaryCheckpoint) -> bool:
    for field_name, value in checkpoint:
        if field_name in {"anchors", "artifact_ids"} or not isinstance(value, tuple):
            continue
        normalized = tuple(_normalize(item) for item in value if isinstance(item, str))
        if len(normalized) != len(set(normalized)):
            return True
    return False


__all__ = ("deterministic_evaluator",)
