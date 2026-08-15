from __future__ import annotations

import pytest
from pydantic import BaseModel

from kedi_summarization_optimizer import (
    CodexModelsSettings,
    EvaluationContract,
    HistoryMessage,
    SemanticJudgement,
    SummarizationInput,
    SummaryCheckpoint,
    codex_runtime,
)


def test_runtime_configuration_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_runtime, "_runtime", None)
    settings = CodexModelsSettings()

    codex_runtime.configure_codex_runtime(settings)

    assert codex_runtime._runtime is None
    assert codex_runtime._settings == settings


@pytest.mark.asyncio
async def test_hard_gate_failure_does_not_spend_a_judge_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = SummarizationInput(
        example_id="secret",
        messages=(HistoryMessage(role="user", content="Do not copy SECRET_CANARY."),),
        evaluation=EvaluationContract(sensitive_values=("SECRET_CANARY",)),
    )
    expected = SummaryCheckpoint(current_objective="Continue safely.")
    output = SummaryCheckpoint(
        current_objective="Continue safely.",
        recent_progress=("Copied SECRET_CANARY.",),
    )

    def fail_if_called() -> None:
        raise AssertionError("Terra judge must not run after a deterministic hard failure.")

    monkeypatch.setattr(codex_runtime, "_get_runtime", fail_if_called)

    result = await codex_runtime.codex_evaluator(inputs, output, expected)

    assert result.hard_pass is False
    assert result.score <= 0.25
    assert result.metrics["sensitive_leaks"] == 1


def test_redaction_removes_every_evaluator_canary() -> None:
    content = "token-A appears before token-B and token-A appears twice"

    redacted = codex_runtime._redact(content, ("token-A", "token-B"))

    assert redacted == (
        "<SENSITIVE_VALUE> appears before <SENSITIVE_VALUE> and <SENSITIVE_VALUE> appears twice"
    )


@pytest.mark.asyncio
async def test_semantic_judge_owns_quality_score_after_hard_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = SummarizationInput(
        example_id="semantic",
        messages=(HistoryMessage(role="user", content="Continue the optimizer."),),
    )
    expected = SummaryCheckpoint(current_objective="Continue the optimizer.")
    judgement = SemanticJudgement(
        operating_rule_fidelity=0.8,
        project_context_fidelity=0.8,
        current_state_fidelity=0.8,
        recent_progress_fidelity=0.8,
        latest_wins_fidelity=0.8,
        grounded=True,
    )

    class _Result(BaseModel):
        output: SemanticJudgement

    class _Judge:
        async def run(self, _prompt: str) -> _Result:
            return _Result(output=judgement)

    class _Runtime:
        judge = _Judge()

    monkeypatch.setattr(codex_runtime, "_get_runtime", _Runtime)

    result = await codex_runtime.codex_evaluator(inputs, expected, expected)

    assert result.hard_pass is True
    assert result.score == pytest.approx(0.8)
    assert result.metrics["deterministic_fidelity_score"] == 1.0
