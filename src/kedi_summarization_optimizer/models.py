"""Stable, provider-neutral models shared by the optimization pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


CheckpointItem = Annotated[str, Field(min_length=1)]


class HistoryMessage(FrozenModel):
    role: Literal["system", "user", "assistant", "tool", "commentary"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class CanonicalAnchor(FrozenModel):
    kind: str = Field(min_length=1)
    id: str = Field(min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationContract(FrozenModel):
    """Deterministic constraints that a semantic judge must not override."""

    forbidden_phrases: tuple[CheckpointItem, ...] = ()
    sensitive_values: tuple[CheckpointItem, ...] = ()

    @model_validator(mode="after")
    def require_unique_non_empty_values(self) -> EvaluationContract:
        for field_name in ("forbidden_phrases", "sensitive_values"):
            values = getattr(self, field_name)
            if len({value.casefold() for value in values}) != len(values):
                raise ValueError(f"{field_name} cannot contain duplicate values.")
        return self


class SummarizationInput(FrozenModel):
    example_id: str = Field(min_length=1)
    messages: tuple[HistoryMessage, ...]
    anchors: tuple[CanonicalAnchor, ...] = ()
    evaluation: EvaluationContract = Field(default_factory=EvaluationContract)
    max_output_chars: int = Field(default=8_000, ge=256)

    @model_validator(mode="after")
    def require_history(self) -> SummarizationInput:
        if not self.messages:
            raise ValueError("A summarization input must contain at least one message.")
        anchor_ids = tuple(anchor.id for anchor in self.anchors)
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("Canonical anchor IDs must be unique within one history.")
        return self


class SummaryCheckpoint(FrozenModel):
    current_objective: CheckpointItem = Field(
        description="The one active objective a fresh agent should continue now."
    )
    operating_rules: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="Current durable user or repository rules; omit superseded rules.",
    )
    project_context: tuple[CheckpointItem, ...] = Field(
        default=(),
        description=(
            "Stable project architecture, semantics, and facts needed for future decisions."
        ),
    )
    constraints: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="Active task or scope constraints that still apply.",
    )
    decisions: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="Effective decisions after applying later corrections and replacements.",
    )
    recent_progress: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="A concise account of recent meaningful work, not a turn-by-turn narrative.",
    )
    current_execution_state: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="Current branch, validation, runtime, or implementation state when known.",
    )
    completed_actions: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="Verified completed work that should not be repeated.",
    )
    resources: tuple[CheckpointItem, ...] = Field(
        default=(),
        description=(
            "Concrete files, URLs, commits, branches, or named resources needed to continue."
        ),
    )
    unresolved_problems: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="Problems that remain open; do not include resolved failures.",
    )
    pending_actions: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="Specific next actions that remain to be performed.",
    )
    lifecycle_outcomes: tuple[CheckpointItem, ...] = Field(
        default=(),
        description="Verified test, build, commit, deployment, or publication outcomes.",
    )
    uncertainties: tuple[CheckpointItem, ...] = Field(
        default=(),
        description=(
            "Material uncertainty explicitly present in the source; do not invent uncertainty."
        ),
    )
    artifact_ids: tuple[CheckpointItem, ...] = Field(
        default=(),
        description=(
            "Exact opaque artifact ref IDs explicitly returned by artifact tools and still needed. "
            "Anchor IDs, tool call IDs, filenames, and inferred IDs are not artifact refs."
        ),
    )
    anchors: tuple[CanonicalAnchor, ...] = Field(
        default=(),
        description="Canonical input anchors copied exactly, with unchanged IDs and payloads.",
    )


class ExampleMetadata(FrozenModel):
    scenario_family: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    labels: dict[str, str] = Field(default_factory=dict)


class HistoryExample(FrozenModel):
    id: str = Field(min_length=1)
    input: SummarizationInput
    expected: SummaryCheckpoint
    metadata: ExampleMetadata

    @model_validator(mode="after")
    def align_identity(self) -> HistoryExample:
        if self.id != self.input.example_id:
            raise ValueError("HistoryExample.id must match input.example_id.")
        expected_text = json.dumps(
            self.expected.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        ).casefold()
        for field_name in ("forbidden_phrases", "sensitive_values"):
            for value in getattr(self.input.evaluation, field_name):
                if value.casefold() in expected_text:
                    raise ValueError(
                        f"HistoryExample.expected cannot contain {field_name} value {value!r}."
                    )
        return self


class DatasetBundle(FrozenModel):
    version: str = Field(min_length=1)
    train: tuple[HistoryExample, ...]
    validation: tuple[HistoryExample, ...]
    heldout: tuple[HistoryExample, ...]

    @model_validator(mode="after")
    def validate_splits(self) -> DatasetBundle:
        if not self.train or not self.validation or not self.heldout:
            raise ValueError("train, validation, and heldout splits must all be non-empty.")
        split_ids = {
            "train": {example.id for example in self.train},
            "validation": {example.id for example in self.validation},
            "heldout": {example.id for example in self.heldout},
        }
        overlaps = (
            split_ids["train"] & split_ids["validation"]
            | split_ids["train"] & split_ids["heldout"]
            | split_ids["validation"] & split_ids["heldout"]
        )
        if overlaps:
            names = ", ".join(sorted(overlaps))
            raise ValueError(f"Dataset split IDs overlap: {names}")
        return self

    def fingerprints(self) -> dict[str, str]:
        return {
            name: _fingerprint_examples(examples)
            for name, examples in (
                ("train", self.train),
                ("validation", self.validation),
                ("heldout", self.heldout),
            )
        }


class CheckpointEvaluation(FrozenModel):
    score: float = Field(ge=0.0, le=1.0)
    hard_pass: bool
    feedback: tuple[str, ...] = ()
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)


class SemanticJudgement(FrozenModel):
    operating_rule_fidelity: float = Field(ge=0.0, le=1.0)
    project_context_fidelity: float = Field(ge=0.0, le=1.0)
    current_state_fidelity: float = Field(ge=0.0, le=1.0)
    recent_progress_fidelity: float = Field(ge=0.0, le=1.0)
    latest_wins_fidelity: float = Field(ge=0.0, le=1.0)
    grounded: bool
    critical_omissions: tuple[str, ...] = ()
    feedback: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return (
            self.operating_rule_fidelity * 0.30
            + self.project_context_fidelity * 0.20
            + self.current_state_fidelity * 0.20
            + self.recent_progress_fidelity * 0.15
            + self.latest_wins_fidelity * 0.15
        )


class OptimizationOutcome(FrozenModel):
    selected_instructions: str
    candidate_id: str
    candidate_fingerprint: str
    best_score: float
    validation_score: float | None = None
    metric_calls: int | None = Field(default=None, ge=0)
    stop_reason: str | None = None
    dataset_fingerprints: dict[str, str]


class CertificationCaseOutcome(FrozenModel):
    example_id: str
    score: float = Field(ge=0.0, le=1.0)
    hard_pass: bool
    feedback: tuple[str, ...] = ()
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)


class CertificationOutcome(FrozenModel):
    accepted: bool
    mean_score: float = Field(ge=0.0, le=1.0)
    minimum_score: float = Field(ge=0.0, le=1.0)
    cases: tuple[CertificationCaseOutcome, ...]


class CampaignOutcome(FrozenModel):
    campaign_id: str
    accepted: bool
    selected_candidate_path: str
    accepted_prompt_path: str | None = None
    optimization_record_path: str
    certification_record_path: str
    optimization: OptimizationOutcome
    certification: CertificationOutcome


def _fingerprint_examples(examples: tuple[HistoryExample, ...]) -> str:
    payload = [example.model_dump(mode="json") for example in examples]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = (
    "CampaignOutcome",
    "CanonicalAnchor",
    "CertificationCaseOutcome",
    "CertificationOutcome",
    "CheckpointEvaluation",
    "DatasetBundle",
    "EvaluationContract",
    "ExampleMetadata",
    "HistoryExample",
    "HistoryMessage",
    "OptimizationOutcome",
    "SemanticJudgement",
    "SummarizationInput",
    "SummaryCheckpoint",
)
