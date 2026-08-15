"""Codex-backed target, judge, and reflection models for optimization campaigns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import RLock
from typing import cast

from codex_auth_helper import CodexResponsesModel, create_codex_responses_model
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_gepa.integrations.pydantic_ai import PydanticAIReflectionModel
from pydantic_gepa.reflection import ReflectionPrompt

from .config import CodexModelRole, CodexModelsSettings
from .evaluation import deterministic_evaluator
from .models import (
    CheckpointEvaluation,
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
You improve a history-summarization system prompt from normalized evaluation evidence. Produce a
concise replacement prompt that improves current-state reconstruction, latest-wins conflict
resolution, durable-rule retention, recent-progress compression, grounding, and secret safety.
Do not solve the benchmark examples or mention their identifiers.
"""


@dataclass(frozen=True, slots=True)
class _CodexRuntime:
    settings: CodexModelsSettings
    summarizer: Agent[None, SummaryCheckpoint]
    judge: Agent[None, SemanticJudgement]
    reflector: PydanticAIReflectionModel[None]


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


__all__ = (
    "codex_evaluator",
    "codex_invoker",
    "configure_codex_runtime",
    "terra_reflect",
)
