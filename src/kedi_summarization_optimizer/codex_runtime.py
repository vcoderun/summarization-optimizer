"""Codex-backed target, judge, and reflection models for optimization campaigns."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import cast

from codex_auth_helper import CodexResponsesModel, create_codex_responses_model
from pydantic import Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_gepa.integrations.pydantic_ai import PydanticAIReflectionModel
from pydantic_gepa.reflection import ReflectionPrompt
from pydantic_gepa.values import SerializableValue

from .config import CodexModelRole, CodexModelsSettings
from .evaluation import deterministic_evaluator
from .models import (
    CheckpointEvaluation,
    FrozenModel,
    SemanticJudgement,
    SummarizationInput,
    SummaryCheckpoint,
)

_SUMMARIZER_SYSTEM = """\
You reconstruct the current working state of a software-engineering conversation for a fresh
agent epoch. Return only the requested structured checkpoint. Preserve effective durable rules,
project facts, current decisions, unresolved work, concise recent progress, and exact canonical
references. Later applicable user instructions supersede earlier conflicting instructions. Never
copy credentials or retain superseded state. Do not narrate the conversation.
"""

_JUDGE_SYSTEM = """\
You evaluate whether a structured checkpoint lets a fresh software-engineering agent continue
without material loss. Judge semantic fidelity rather than wording. The latest applicable user
instruction wins over conflicting older instructions. Penalize invented facts, stale decisions,
missing durable rules, missing project context, and misleading lifecycle state. Return only the
requested structured judgement.
"""

_REFLECTOR_SYSTEM = """\
You improve a production history-summarization system prompt from normalized evaluation evidence.
The runtime model receives only ordered_history, canonical_anchors, max_output_chars, and a
separately enforced structured output schema. Evaluation contracts, expected checkpoints, dataset
labels, example IDs, scores, hard-pass gates, and judge feedback exist only in the optimizer. Never
claim that the runtime model can see those optimizer-only values. Do not restate the JSON schema,
key order, benchmark structure, or example-specific wording. Produce a compact replacement prompt,
preferably below 4,000 characters, that generalizes across real Kedi engineering conversations:
current-state reconstruction, latest-wins conflict resolution, durable-rule retention, lifecycle
truth, relevant recent progress, aggressive noise compression, grounding, and secret safety.
"""

_PROPOSER_SYSTEM = """\
Write one compact, production-safe replacement policy for a history summarizer. Return only
context-independent imperative instructions that remain correct for unseen histories. Every
sentence must tell the production summarizer what to do. Do not analyze, quote, paraphrase, or
describe the current candidate, observed failures, aggregate evidence, or a current history. Do not
include observations, counts, scores, metric names, or claims that some state is currently absent.
Return 5-16 rules in the structured rules field. Make each rule exactly one sentence and begin it
with an imperative verb or a reusable If/When condition.
Express absence handling conditionally as a reusable rule; never assert that information is absent
from the current input. A proposal containing a present-state observation will be rejected.
The production model receives ordered_history, canonical_anchors, max_output_chars, and a separately
enforced output schema. Never mention optimizer internals, datasets, examples, judges, expected
outputs, validation gates, or specific Kedi fixtures and syntax. Do not restate the output schema.
Generalize only into reusable principles for latest-applicable instructions, evidence and lifecycle
truth, durable rules, unresolved work, relevant progress, reference validity, secret safety, and
compression.
"""

MAX_PROPOSAL_CHARS = 4_500
_SAFE_DIAGNOSTIC_KEYS = frozenset(
    {
        "active_state_score",
        "deterministic_fidelity_score",
        "episodic_score",
        "forbidden_leaks",
        "hard_pass",
        "invented_anchors",
        "invented_artifacts",
        "judge_critical_omissions",
        "judge_grounded",
        "judge_latest_wins",
        "judge_score",
        "mismatched_anchors",
        "missing_anchors",
        "missing_artifacts",
        "output_chars",
        "reference_score",
        "semantic_score",
        "sensitive_leaks",
    }
)
_FORBIDDEN_PROPOSAL_FRAGMENTS = (
    "```",
    "> skills:",
    "> import:",
    "aggregate evidence",
    "aggregate failure",
    "are available in",
    "are retained in this",
    "current active work",
    "current candidate",
    "current canonical anchors",
    "current governing",
    "current_instructions",
    "case_name",
    "dataset",
    "diagnostic",
    "docs/language/templates.md",
    "evaluated case",
    "evaluation evidence",
    "evaluation constraints",
    "evaluation.forbidden",
    "evaluation.sensitive",
    "example_id",
    "expected checkpoint",
    "expected output",
    "failure category",
    "hard-pass",
    "kedi_http",
    "lazyadapter",
    "luna high",
    "mean score",
    "metric name",
    "next state update",
    "no project-specific",
    "no sensitive values",
    "observed failure",
    "recorded failure",
    "run_main",
    "skill-creator",
    "solve_knapsack",
    "successful case",
    "terra high",
    "there are no",
    "this exact schema",
    "until then",
    "validation set",
    "virtual python",
)
_OBSERVATIONAL_ABSENCE = re.compile(r"\b(?:no|none)\b", re.IGNORECASE)
_MULTIPLE_SENTENCES = re.compile(r"[.!?]\s+\S")
_IMPERATIVE_RULE_PREFIXES = (
    "apply ",
    "avoid ",
    "carry ",
    "classify ",
    "compress ",
    "distinguish ",
    "do ",
    "exclude ",
    "if ",
    "include ",
    "keep ",
    "limit ",
    "maintain ",
    "mark ",
    "never ",
    "omit ",
    "prefer ",
    "preserve ",
    "prioritize ",
    "reconstruct ",
    "record ",
    "reject ",
    "remove ",
    "resolve ",
    "retain ",
    "treat ",
    "use ",
    "verify ",
    "when ",
)


class PromptProposal(FrozenModel):
    """Validated mutation text that can be shipped into the runtime unchanged."""

    rules: tuple[str, ...] = Field(min_length=5, max_length=16)

    @model_validator(mode="after")
    def require_provider_neutral_instructions(self) -> PromptProposal:
        for rule in self.rules:
            normalized = rule.strip()
            folded = normalized.casefold()
            forbidden = tuple(
                fragment
                for fragment in _FORBIDDEN_PROPOSAL_FRAGMENTS
                if fragment.casefold() in folded
            )
            if forbidden:
                raise ValueError(
                    "Proposal contains optimizer-only or case-specific material: "
                    + ", ".join(forbidden)
                )
            if _OBSERVATIONAL_ABSENCE.search(normalized):
                raise ValueError(
                    "Proposal contains a present-state absence claim instead of a reusable rule."
                )
            if "\n" in normalized or _MULTIPLE_SENTENCES.search(normalized):
                raise ValueError("Each proposal rule must be one sentence.")
            if not folded.startswith(_IMPERATIVE_RULE_PREFIXES):
                raise ValueError("Each proposal rule must be imperative or conditional.")

        if not 300 <= len(self.instructions) <= MAX_PROPOSAL_CHARS:
            raise ValueError(
                f"Combined proposal must contain between 300 and {MAX_PROPOSAL_CHARS} characters."
            )
        return self

    @property
    def instructions(self) -> str:
        return "\n".join(f"- {rule.strip()}" for rule in self.rules)


@dataclass(frozen=True, slots=True)
class _CodexRuntime:
    settings: CodexModelsSettings
    summarizer: Agent[None, SummaryCheckpoint]
    judge: Agent[None, SemanticJudgement]
    reflector: PydanticAIReflectionModel[None]
    proposer: Agent[None, PromptProposal]


_lock = RLock()
_settings = CodexModelsSettings()
_runtime: _CodexRuntime | None = None


def configure_codex_runtime(settings: CodexModelsSettings) -> None:
    """Select one immutable model profile before an optimization task starts."""

    global _runtime, _settings
    with _lock:
        if settings == _settings:
            return
        _settings = settings
        _runtime = None


async def codex_invoker(
    instructions: str,
    inputs: SummarizationInput,
) -> SummaryCheckpoint:
    payload = {
        "messages": [message.model_dump(mode="json") for message in inputs.messages],
        "canonical_anchors": [anchor.model_dump(mode="json") for anchor in inputs.anchors],
        "max_output_chars": inputs.max_output_chars,
    }
    prompt = (
        "Reconstruct the effective state from this ordered history. The candidate instructions "
        "are authoritative for extraction strategy, while the structured output schema defines "
        "the required sections.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    result = await _get_runtime().summarizer.run(prompt, instructions=instructions)
    return result.output


async def codex_evaluator(
    inputs: SummarizationInput,
    output: SummaryCheckpoint,
    expected: SummaryCheckpoint,
) -> CheckpointEvaluation:
    deterministic = deterministic_evaluator(inputs, output, expected)
    if not deterministic.hard_pass:
        return deterministic.model_copy(update={"score": min(deterministic.score, 0.25)})

    sensitive_values = inputs.evaluation.sensitive_values
    source_messages = [
        {
            **message.model_dump(mode="json"),
            "content": _redact(message.content, sensitive_values),
        }
        for message in inputs.messages
    ]
    payload = {
        "ordered_history": source_messages,
        "canonical_anchors": [anchor.model_dump(mode="json") for anchor in inputs.anchors],
        "expected_checkpoint": expected.model_dump(mode="json"),
        "actual_checkpoint": output.model_dump(mode="json"),
    }
    judgement = (
        await _get_runtime().judge.run(
            "Evaluate this checkpoint pair. Do not reward exact wording; assess whether the actual "
            "checkpoint preserves the expected effective state and excludes stale state.\n\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    ).output

    semantic_hard_pass = (
        judgement.grounded
        and not judgement.critical_omissions
        and judgement.latest_wins_fidelity >= 0.8
        and judgement.score >= 0.75
    )
    metrics = {
        **deterministic.metrics,
        "deterministic_fidelity_score": deterministic.score,
        "judge_score": judgement.score,
        "judge_grounded": judgement.grounded,
        "judge_critical_omissions": len(judgement.critical_omissions),
        "judge_latest_wins": judgement.latest_wins_fidelity,
    }
    return CheckpointEvaluation(
        score=judgement.score,
        hard_pass=semantic_hard_pass,
        feedback=deterministic.feedback + judgement.feedback + judgement.critical_omissions,
        metrics=metrics,
    )


def terra_reflect(prompt: ReflectionPrompt) -> str:
    """Keep the checkpointed GEPA config pickle-safe while using native Pydantic reflection."""

    return _get_runtime().reflector(prompt)


def terra_propose(
    candidate: dict[str, str],
    reflective_dataset: Mapping[str, Sequence[Mapping[str, SerializableValue]]],
    components_to_update: list[str],
) -> dict[str, str]:
    """Propose bounded runtime instructions from sanitized aggregate evidence."""

    proposals = dict(candidate)
    for component in components_to_update:
        current = candidate.get(component)
        if current is None:
            raise ValueError(f"Missing candidate component: {component}")
        evidence = _sanitize_reflection_evidence(reflective_dataset.get(component, ()))
        result = _get_runtime().proposer.run_sync(
            json.dumps(
                {
                    "current_instructions": current,
                    "aggregate_failure_evidence": evidence,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        proposals[component] = result.output.instructions
    return proposals


def _get_runtime() -> _CodexRuntime:
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = _build_runtime(_settings)
        return _runtime


def _build_runtime(settings: CodexModelsSettings) -> _CodexRuntime:
    summarizer_model = _create_model(settings.summarizer, _SUMMARIZER_SYSTEM)
    judge_model = _create_model(settings.judge, _JUDGE_SYSTEM)
    reflector_model = _create_model(settings.reflector, _REFLECTOR_SYSTEM)
    return _CodexRuntime(
        settings=settings,
        summarizer=Agent(
            summarizer_model,
            output_type=SummaryCheckpoint,
            name="kedi_summary_checkpoint_writer",
            retries=2,
        ),
        judge=Agent(
            judge_model,
            output_type=SemanticJudgement,
            name="kedi_summary_checkpoint_judge",
            retries=2,
        ),
        reflector=PydanticAIReflectionModel.from_model(
            reflector_model,
            retries=2,
        ),
        proposer=Agent(
            _create_model(settings.reflector, _PROPOSER_SYSTEM),
            output_type=PromptProposal,
            name="kedi_summary_prompt_proposer",
            retries=4,
        ),
    )


def _create_model(role: CodexModelRole, instructions: str) -> CodexResponsesModel:
    settings = cast(
        "OpenAIResponsesModelSettings",
        {
            "openai_reasoning_effort": role.effort,
            "openai_reasoning_summary": "concise",
            "openai_store": False,
        },
    )
    return create_codex_responses_model(
        role.model_id,
        instructions=instructions,
        settings=settings,
    )


def _redact(content: str, sensitive_values: tuple[str, ...]) -> str:
    for value in sensitive_values:
        content = content.replace(value, "<SENSITIVE_VALUE>")
    return content


def _sanitize_reflection_evidence(
    records: Sequence[Mapping[str, SerializableValue]],
) -> dict[str, SerializableValue]:
    scores: list[float] = []
    successful_cases = 0
    observed_success_values = 0
    failure_categories: dict[str, int] = {}
    feedback_categories: dict[str, int] = {}
    diagnostic_values: dict[str, list[float]] = {}

    for record in records:
        score = record.get("score")
        if isinstance(score, int | float) and not isinstance(score, bool):
            scores.append(float(score))

        success = record.get("success")
        if isinstance(success, bool):
            observed_success_values += 1
            successful_cases += int(success)

        failure_category = record.get("failure_category")
        if isinstance(failure_category, str):
            category = _safe_failure_category(failure_category)
            failure_categories[category] = failure_categories.get(category, 0) + 1

        feedback = record.get("metric_feedback")
        for category in _feedback_category_names(feedback):
            feedback_categories[category] = feedback_categories.get(category, 0) + 1

        _collect_safe_diagnostics(record.get("metric_side_info"), diagnostic_values)

    evidence: dict[str, SerializableValue] = {"evaluated_cases": len(records)}
    if scores:
        evidence["mean_score"] = round(sum(scores) / len(scores), 6)
    if observed_success_values:
        evidence["successful_cases"] = successful_cases
    if failure_categories:
        evidence["failure_categories"] = dict(sorted(failure_categories.items()))
    if feedback_categories:
        evidence["feedback_categories"] = dict(sorted(feedback_categories.items()))
    if diagnostic_values:
        evidence["mean_diagnostics"] = {
            key: round(sum(values) / len(values), 6)
            for key, values in sorted(diagnostic_values.items())
        }
    return evidence


def _collect_safe_diagnostics(
    value: object,
    collected: dict[str, list[float]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key)
            if normalized_key in _SAFE_DIAGNOSTIC_KEYS and isinstance(child, bool | int | float):
                collected.setdefault(normalized_key, []).append(float(child))
            else:
                _collect_safe_diagnostics(child, collected)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for child in value:
            _collect_safe_diagnostics(child, collected)


def _feedback_category_names(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(
            category for child in value.values() for category in _feedback_category_names(child)
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(category for child in value for category in _feedback_category_names(child))
    if not isinstance(value, str):
        return ()

    folded = value.casefold()
    categories = tuple(
        category
        for category, fragments in (
            ("secret_safety", ("credential", "secret", "sensitive")),
            ("latest_applicable_state", ("latest", "stale", "superseded", "replaced")),
            ("canonical_anchor_fidelity", ("anchor",)),
            ("reference_fidelity", ("artifact", "reference", "resource")),
            ("active_state", ("objective", "pending", "unresolved", "active state")),
            ("durable_semantics", ("rule", "constraint", "decision", "project context")),
            ("lifecycle_truth", ("lifecycle", "completed", "progress")),
            ("deduplication", ("duplicate", "redundan")),
            ("compression", ("max_output", "too long", "verbose", "compact")),
            ("grounding", ("invent", "hallucin", "ground")),
        )
        if any(fragment in folded for fragment in fragments)
    )
    return categories or ("semantic_fidelity",)


def _safe_failure_category(value: str) -> str:
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"cancelled", "constraint", "quality", "runtime", "timeout", "tool"}:
        return normalized
    return "other"


__all__ = (
    "codex_evaluator",
    "codex_invoker",
    "configure_codex_runtime",
    "terra_propose",
    "terra_reflect",
)
