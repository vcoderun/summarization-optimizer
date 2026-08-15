from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

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


def test_prompt_proposal_rejects_optimizer_only_and_oversized_text() -> None:
    valid = (
        "Reconstruct current state from ordered history using explicit evidence and applicable "
        "instructions.",
        "Preserve durable rules, active work, verified lifecycle outcomes, useful resources, and "
        "live references.",
        "Resolve conflicting statements within their shared subject and scope by applying the "
        "latest instruction.",
        "Retain uncertainty when evidence is incomplete and the gap affects continuation.",
        "Remove stale, resolved, unrelated, duplicated, and sensitive content.",
        "Keep the checkpoint compact enough for a fresh agent to continue the active work.",
    )

    expected = "\n".join(f"- {rule}" for rule in valid)
    assert codex_runtime.PromptProposal(rules=valid).instructions == expected
    with pytest.raises(ValidationError, match="optimizer-only"):
        codex_runtime.PromptProposal(rules=(*valid[:-1], "Use evaluation constraints."))
    with pytest.raises(ValidationError, match="optimizer-only"):
        codex_runtime.PromptProposal(rules=(*valid[:-1], "Improve the mean score."))
    with pytest.raises(ValidationError, match="absence claim"):
        codex_runtime.PromptProposal(rules=(*valid[:-1], "Retain no durable state."))
    with pytest.raises(ValidationError, match="one sentence"):
        codex_runtime.PromptProposal(
            rules=(*valid[:-1], "Preserve active work. Remove stale work.")
        )
    with pytest.raises(ValidationError, match="imperative"):
        codex_runtime.PromptProposal(rules=(*valid[:-1], "Active work should remain available."))
    with pytest.raises(ValidationError, match="between 300"):
        codex_runtime.PromptProposal(rules=tuple("Preserve " + "x" * 900 for _ in range(5)))


def test_reflection_evidence_excludes_raw_benchmark_material() -> None:
    sanitized = codex_runtime._sanitize_reflection_evidence(
        (
            {
                "case_name": "approval-lifecycle",
                "inputs": {"messages": ["PRIVATE_HISTORY"]},
                "expected_output": {"current_objective": "PRIVATE_EXPECTED"},
                "metadata": {"scenario_family": "PRIVATE_FAMILY"},
                "actual_output": {"current_objective": "PRIVATE_ACTUAL"},
                "traces": ["PRIVATE_TRACE"],
                "score": 0.25,
                "success": False,
                "failure_category": "quality",
                "metric_feedback": {"checkpoint_quality": "Missing canonical anchors."},
                "metric_side_info": {
                    "checkpoint_quality": {
                        "missing_anchors": 2,
                        "example_id": "PRIVATE_ID",
                    }
                },
            },
        )
    )
    rendered = str(sanitized)

    assert sanitized["evaluated_cases"] == 1
    assert sanitized["mean_score"] == 0.25
    assert sanitized["successful_cases"] == 0
    assert sanitized["feedback_categories"] == {"canonical_anchor_fidelity": 1}
    assert sanitized["mean_diagnostics"] == {"missing_anchors": 2.0}
    for private in (
        "PRIVATE_HISTORY",
        "PRIVATE_EXPECTED",
        "PRIVATE_FAMILY",
        "PRIVATE_ACTUAL",
        "PRIVATE_TRACE",
        "PRIVATE_ID",
        "case_name",
        "inputs",
        "expected_output",
    ):
        assert private not in rendered


def test_terra_proposer_sends_only_sanitized_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = (
        "Reconstruct current project state using explicit evidence and later applicable "
        "instructions.",
        "Retain durable rules and unresolved work while preserving verified lifecycle outcomes.",
        "Exclude stale, unrelated, invalidated, duplicated, and sensitive material.",
        "Resolve conflicting statements only within their shared subject and scope.",
        "Distinguish successful completion from failed, cancelled, or merely attempted work.",
        "Preserve only references that remain necessary for continuation.",
    )
    instructions = "\n".join(f"- {rule}" for rule in rules)

    class _ProposalResult(BaseModel):
        output: codex_runtime.PromptProposal

    class _Proposer:
        prompt: str | None = None

        def run_sync(self, prompt: str) -> _ProposalResult:
            self.prompt = prompt
            return _ProposalResult(output=codex_runtime.PromptProposal(rules=rules))

    proposer = _Proposer()

    class _Runtime:
        def __init__(self, runtime_proposer: _Proposer) -> None:
            self.proposer = runtime_proposer

    runtime = _Runtime(proposer)
    monkeypatch.setattr(codex_runtime, "_get_runtime", lambda: runtime)

    result = codex_runtime.terra_propose(
        {"summarizer_instructions": "seed"},
        {
            "summarizer_instructions": [
                {
                    "inputs": {"messages": ["PRIVATE_HISTORY"]},
                    "expected_output": "PRIVATE_EXPECTED",
                    "score": 0.4,
                    "success": False,
                    "failure_category": "quality",
                    "metric_feedback": {"quality": "Missing active state."},
                }
            ]
        },
        ["summarizer_instructions"],
    )

    assert result == {"summarizer_instructions": instructions}
    assert proposer.prompt is not None
    assert "PRIVATE_HISTORY" not in proposer.prompt
    assert "PRIVATE_EXPECTED" not in proposer.prompt
    assert "active_state" in proposer.prompt
    assert "Missing active state." not in proposer.prompt


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
