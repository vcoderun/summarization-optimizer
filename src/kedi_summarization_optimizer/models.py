"""Stable, provider-neutral models shared by the optimization pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HistoryMessage(FrozenModel):
    role: Literal["system", "user", "assistant", "tool", "commentary"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class CanonicalAnchor(FrozenModel):
    kind: str = Field(min_length=1)
    id: str = Field(min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class SummarizationInput(FrozenModel):
    example_id: str = Field(min_length=1)
    messages: tuple[HistoryMessage, ...]
    anchors: tuple[CanonicalAnchor, ...] = ()
    max_output_chars: int = Field(default=8_000, ge=256)

    @model_validator(mode="after")
    def require_history(self) -> SummarizationInput:
        if not self.messages:
            raise ValueError("A summarization input must contain at least one message.")
        return self


class SummaryCheckpoint(FrozenModel):
    current_objective: str
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    completed_actions: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    unresolved_problems: tuple[str, ...] = ()
    pending_actions: tuple[str, ...] = ()
    lifecycle_outcomes: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    anchors: tuple[CanonicalAnchor, ...] = ()


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
    "ExampleMetadata",
    "HistoryExample",
    "HistoryMessage",
    "OptimizationOutcome",
    "SummarizationInput",
    "SummaryCheckpoint",
)
