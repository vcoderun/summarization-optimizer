from __future__ import annotations

import pytest
from pydantic import ValidationError

from kedi_summarization_optimizer.evaluation import deterministic_evaluator
from kedi_summarization_optimizer.models import (
    CanonicalAnchor,
    EvaluationContract,
    ExampleMetadata,
    HistoryExample,
    HistoryMessage,
    SummarizationInput,
    SummaryCheckpoint,
)


def test_perfect_checkpoint_passes_all_deterministic_gates() -> None:
    inputs, expected = _case()

    result = deterministic_evaluator(inputs, expected, expected)

    assert result.hard_pass is True
    assert result.score == 1.0
    assert result.feedback == ()
    assert result.metrics["semantic_score"] == 1.0
    assert result.metrics["active_state_score"] == 1.0


def test_paraphrase_affects_quality_without_becoming_a_safety_failure() -> None:
    inputs, expected = _case()
    output = expected.model_copy(
        update={"project_context": ("The repository implements the Kedi language.",)}
    )

    result = deterministic_evaluator(inputs, output, expected)

    assert result.hard_pass is True
    assert 0.0 < result.score < 1.0


@pytest.mark.parametrize(
    ("update", "feedback"),
    [
        (
            {"operating_rules": ("Open submodule pull requests.",)},
            "superseded or explicitly forbidden",
        ),
        (
            {"recent_progress": ("Temporary credential: SYNTHETIC_SECRET_91",)},
            "exposes sensitive values",
        ),
        ({"anchors": ()}, "omits required canonical anchors"),
        (
            {
                "anchors": (
                    CanonicalAnchor(
                        kind="operating_rule",
                        id="rule-submodules",
                        payload={"text": "Use pull requests."},
                    ),
                )
            },
            "changes canonical anchor payloads",
        ),
        (
            {
                "anchors": (
                    CanonicalAnchor(
                        kind="operating_rule",
                        id="rule-submodules",
                        payload={"text": "Push submodules directly to main."},
                    ),
                    CanonicalAnchor(kind="decision", id="invented", payload={"text": "No."}),
                )
            },
            "invents canonical anchors",
        ),
        ({"artifact_ids": ()}, "omits required artifact references"),
        (
            {"artifact_ids": ("tool_call_result_7", "tool_call_result_999")},
            "invents artifact references",
        ),
        (
            {"recent_progress": ("Added evaluator tests.", " added  evaluator tests. ")},
            "duplicate entries",
        ),
    ],
)
def test_safety_and_reference_failures_are_hard_gates(
    update: dict[str, object],
    feedback: str,
) -> None:
    inputs, expected = _case()

    result = deterministic_evaluator(inputs, expected.model_copy(update=update), expected)

    assert result.hard_pass is False
    assert any(feedback in item for item in result.feedback)


def test_output_budget_is_measured_on_the_serialized_checkpoint() -> None:
    inputs, expected = _case(max_output_chars=256)
    output = expected.model_copy(update={"recent_progress": ("x" * 300,)})

    result = deterministic_evaluator(inputs, output, expected)

    assert result.hard_pass is False
    assert result.metrics["output_chars"] > 256
    assert any("max_output_chars" in item for item in result.feedback)


def test_contract_and_checkpoint_reject_ambiguous_values() -> None:
    with pytest.raises(ValidationError, match="duplicate values"):
        EvaluationContract(forbidden_phrases=("old rule", "old rule"))

    with pytest.raises(ValidationError, match="at least 1 character"):
        SummaryCheckpoint(current_objective="")

    with pytest.raises(ValidationError, match="Canonical anchor IDs must be unique"):
        SummarizationInput(
            example_id="duplicate-anchor",
            messages=(HistoryMessage(role="user", content="Continue."),),
            anchors=(
                CanonicalAnchor(kind="rule", id="same", payload={}),
                CanonicalAnchor(kind="rule", id="same", payload={}),
            ),
        )

    inputs, expected = _case()
    with pytest.raises(ValidationError, match="expected cannot contain forbidden_phrases"):
        HistoryExample(
            id=inputs.example_id,
            input=inputs,
            expected=expected.model_copy(
                update={"operating_rules": ("Open submodule pull requests.",)}
            ),
            metadata=ExampleMetadata(
                scenario_family="latest-wins",
                generator_version="test-v1",
            ),
        )


def _case(*, max_output_chars: int = 8_000) -> tuple[SummarizationInput, SummaryCheckpoint]:
    anchor = CanonicalAnchor(
        kind="operating_rule",
        id="rule-submodules",
        payload={"text": "Push submodules directly to main."},
    )
    inputs = SummarizationInput(
        example_id="latest-wins",
        messages=(
            HistoryMessage(role="user", content="Open pull requests for submodule changes."),
            HistoryMessage(role="assistant", content="I will open pull requests."),
            HistoryMessage(
                role="user",
                content="Do not open submodule PRs. Push submodule changes directly to main.",
            ),
        ),
        anchors=(anchor,),
        evaluation=EvaluationContract(
            forbidden_phrases=("Open submodule pull requests.",),
            sensitive_values=("SYNTHETIC_SECRET_91",),
        ),
        max_output_chars=max_output_chars,
    )
    expected = SummaryCheckpoint(
        current_objective="Continue the summarization optimizer implementation.",
        operating_rules=("Push submodule changes directly to main; do not open PRs.",),
        project_context=("Kedi is a typed orchestration language.",),
        decisions=("Use Luna high for target summarization.",),
        recent_progress=("Added evaluator tests.",),
        current_execution_state=("The optimizer pipeline is on main.",),
        pending_actions=("Build the adversarial dataset.",),
        lifecycle_outcomes=("The offline pipeline tests pass.",),
        artifact_ids=("tool_call_result_7",),
        anchors=(anchor,),
    )
    return inputs, expected
