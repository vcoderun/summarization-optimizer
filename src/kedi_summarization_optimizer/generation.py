"""Deterministic generation of adversarial high-context history datasets."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import (
    CanonicalAnchor,
    DatasetBundle,
    EvaluationContract,
    ExampleMetadata,
    FrozenModel,
    HistoryExample,
    HistoryMessage,
    SummarizationInput,
    SummaryCheckpoint,
)

GENERATOR_VERSION = "deterministic-synthetic-v1"
DEFAULT_DATASET_VERSION = "synthetic-v1"
DEFAULT_SEED = 20_260_815
DEFAULT_TARGET_HISTORY_CHARS = 24_000

SplitName = Literal["train", "validation", "heldout"]
MessageRole = Literal["system", "user", "assistant", "tool", "commentary"]


class SyntheticDatasetConfig(FrozenModel):
    """Reproducible controls for one generated dataset release."""

    version: str = Field(default=DEFAULT_DATASET_VERSION, min_length=1)
    seed: int = DEFAULT_SEED
    target_history_chars: int = Field(default=DEFAULT_TARGET_HISTORY_CHARS, ge=4_000)


@dataclass(frozen=True)
class _Scenario:
    family: str
    split: SplitName
    build: Callable[[int], HistoryExample]


def generate_synthetic_dataset(
    config: SyntheticDatasetConfig | None = None,
) -> DatasetBundle:
    """Generate a stable dataset whose expected state never depends on an LLM."""

    resolved = config or SyntheticDatasetConfig()
    splits: dict[SplitName, list[HistoryExample]] = {
        "train": [],
        "validation": [],
        "heldout": [],
    }
    for scenario in _SCENARIOS:
        for variant in range(2):
            base = scenario.build(variant)
            example = base.model_copy(
                update={
                    "input": base.input.model_copy(
                        update={
                            "messages": _expand_history(
                                base.input.messages,
                                example_id=base.id,
                                seed=resolved.seed,
                                target_chars=resolved.target_history_chars,
                            )
                        }
                    ),
                    "metadata": base.metadata.model_copy(
                        update={
                            "labels": {
                                **base.metadata.labels,
                                "seed": str(resolved.seed),
                                "target_history_chars": str(resolved.target_history_chars),
                                "variant": str(variant + 1),
                            }
                        }
                    ),
                }
            )
            splits[scenario.split].append(example)

    dataset = DatasetBundle(
        version=resolved.version,
        train=tuple(splits["train"]),
        validation=tuple(splits["validation"]),
        heldout=tuple(splits["heldout"]),
    )
    _validate_generated_dataset(dataset, target_chars=resolved.target_history_chars)
    return dataset


def write_synthetic_dataset(
    path: str | Path,
    dataset: DatasetBundle,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish a generated dataset without silent replacement."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {destination}")

    payload = dataset.model_dump_json(indent=2) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def synthetic_dataset_summary(dataset: DatasetBundle) -> dict[str, object]:
    """Return publication-safe provenance and structural statistics."""

    split_examples = {
        "train": dataset.train,
        "validation": dataset.validation,
        "heldout": dataset.heldout,
    }
    return {
        "version": dataset.version,
        "generator_version": GENERATOR_VERSION,
        "fingerprints": dataset.fingerprints(),
        "splits": {
            split: {
                "examples": len(examples),
                "families": sorted({item.metadata.scenario_family for item in examples}),
                "history_chars": sum(_history_chars(item) for item in examples),
                "minimum_history_chars": min(_history_chars(item) for item in examples),
            }
            for split, examples in split_examples.items()
        },
    }


def _example(
    *,
    family: str,
    variant: int,
    messages: tuple[HistoryMessage, ...],
    expected: SummaryCheckpoint,
    anchors: tuple[CanonicalAnchor, ...],
    forbidden: tuple[str, ...] = (),
    sensitive: tuple[str, ...] = (),
    signal: str,
) -> HistoryExample:
    example_id = f"{family}-v{variant + 1}"
    return HistoryExample(
        id=example_id,
        input=SummarizationInput(
            example_id=example_id,
            messages=messages,
            anchors=anchors,
            evaluation=EvaluationContract(
                forbidden_phrases=forbidden,
                sensitive_values=sensitive,
            ),
            max_output_chars=5_000,
        ),
        expected=expected.model_copy(update={"anchors": anchors}),
        metadata=ExampleMetadata(
            scenario_family=family,
            generator_version=GENERATOR_VERSION,
            labels={"signal": signal},
        ),
    )


def _message(
    role: MessageRole,
    content: str,
    *,
    name: str | None = None,
    call: str | None = None,
) -> HistoryMessage:
    return HistoryMessage(role=role, content=content, name=name, tool_call_id=call)


def _anchor(kind: str, anchor_id: str, text: str) -> CanonicalAnchor:
    return CanonicalAnchor(kind=kind, id=anchor_id, payload={"text": text})


def _latest_wins(variant: int) -> HistoryExample:
    options = (
        (
            "Open pull requests for editor submodules.",
            "Push editor submodule changes directly to main and pin their exact revisions.",
            "Grammar and editor commits were pushed to their main branches.",
            "Update the root revisions and run release checks.",
        ),
        (
            "Use Luna medium for history summarization.",
            "Use Luna high for summarization and Terra high for reflection and judging.",
            "The campaign model roles were updated.",
            "Regenerate the campaign fingerprint and validate checkpoint resume.",
        ),
    )
    old_rule, current_rule, progress, pending = options[variant]
    anchors = (
        _anchor("operating_rule", f"latest-rule-{variant}", current_rule),
        _anchor("pending_action", f"latest-pending-{variant}", pending),
    )
    return _example(
        family="latest-wins",
        variant=variant,
        messages=(
            _message("user", old_rule),
            _message("assistant", f"Understood: {old_rule}"),
            _message("user", f"Correction. {current_rule}"),
            _message("assistant", progress),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            operating_rules=(current_rule,),
            recent_progress=(progress,),
            pending_actions=(pending,),
        ),
        anchors=anchors,
        forbidden=(old_rule,),
        signal="superseded-rule",
    )


def _resolved_failure(variant: int) -> HistoryExample:
    options = (
        (
            "Python 3.14 compatibility is failing because AgentRetries cannot be imported.",
            "Replaced the stale import and added a compatibility regression test.",
            "make prod passed on Python 3.11 through 3.14 with 100% coverage.",
            "Commit the validated compatibility fix.",
        ),
        (
            "The documentation example parser rejects the skills directive.",
            "Corrected the fenced example and added it to documentation validation.",
            "Documentation validation and the strict site build passed.",
            "Publish the corrected documentation revision.",
        ),
    )
    failure, fix, outcome, pending = options[variant]
    anchors = (
        _anchor("lifecycle_outcome", f"resolved-outcome-{variant}", outcome),
        _anchor("pending_action", f"resolved-pending-{variant}", pending),
    )
    return _example(
        family="resolved-failure",
        variant=variant,
        messages=(
            _message("assistant", failure),
            _message("assistant", fix),
            _message("tool", outcome, name="shell", call=f"resolved-call-{variant}"),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            recent_progress=(fix,),
            completed_actions=(fix,),
            pending_actions=(pending,),
            lifecycle_outcomes=(outcome,),
        ),
        anchors=anchors,
        forbidden=(failure,),
        signal="resolved-lifecycle",
    )


def _parallel_tools(variant: int) -> HistoryExample:
    options = (
        (
            "src/kedi/runtime.py",
            "tests/test_runtime.py",
            "Runtime validation passed; the test file still needs a cancellation assertion.",
        ),
        (
            "src/autobench/reporting.py",
            "tests/test_reporting_and_exports.py",
            "Report projection passed; the export test still needs an aggregate-budget assertion.",
        ),
    )
    source_file, test_file, state = options[variant]
    pending = f"Add the missing assertion to {test_file}."
    anchors = (
        _anchor("resource", f"parallel-source-{variant}", source_file),
        _anchor("pending_action", f"parallel-pending-{variant}", pending),
    )
    return _example(
        family="parallel-tools",
        variant=variant,
        messages=(
            _message("assistant", "I am reading the source and test files in parallel."),
            _message(
                "assistant", json.dumps({"path": source_file}), name="read_file", call="read-source"
            ),
            _message(
                "assistant", json.dumps({"path": test_file}), name="read_file", call="read-test"
            ),
            _message("tool", f"test content from {test_file}", name="read_file", call="read-test"),
            _message(
                "tool", f"source content from {source_file}", name="read_file", call="read-source"
            ),
            _message("assistant", state),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            recent_progress=(state,),
            resources=(source_file, test_file),
            pending_actions=(pending,),
        ),
        anchors=anchors,
        signal="parallel-tool-order",
    )


def _approval_lifecycle(variant: int) -> HistoryExample:
    options = (
        (
            "apply_patch",
            "Edit only src/kedi/history.py; do not touch runtime.py.",
            "The scoped history patch was approved and applied.",
            "Run the history regression tests.",
        ),
        (
            "git push",
            "Push only the tree-sitter commit to main; do not push root agent changes.",
            "The scoped tree-sitter push was approved and completed.",
            "Update the exact tree-sitter revision in the editor extension.",
        ),
    )
    tool, edited_scope, outcome, pending = options[variant]
    denied = f"A broader {tool} request was denied and must not be treated as executed."
    pending_approval = f"A separate {tool} follow-up remains pending approval and has not run."
    anchors = (
        _anchor("decision", f"approval-scope-{variant}", edited_scope),
        _anchor("pending_action", f"approval-pending-{variant}", pending),
    )
    return _example(
        family="approval-lifecycle",
        variant=variant,
        messages=(
            _message("assistant", f"Requesting approval for {tool} with broad arguments."),
            _message("tool", "decision=edit", name="approval", call="approval-edit"),
            _message("user", edited_scope),
            _message("tool", outcome, name=tool, call="approved-call"),
            _message(
                "tool",
                "decision=deny for the broader follow-up",
                name="approval",
                call="approval-deny",
            ),
            _message(
                "tool",
                "decision=pending for a separate follow-up",
                name="approval",
                call="approval-pending",
            ),
            _message("assistant", denied),
            _message("assistant", pending_approval),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            constraints=(edited_scope,),
            decisions=(edited_scope,),
            recent_progress=(outcome,),
            completed_actions=(outcome,),
            unresolved_problems=(pending_approval,),
            pending_actions=(pending,),
        ),
        anchors=anchors,
        forbidden=(f"The broader {tool} request was executed.",),
        signal="approval-edit-deny",
    )


def _artifact_lifecycle(variant: int) -> HistoryExample:
    options = (
        ("tool_call_result_41", "deployment deploy-814", "Verify the rollback window."),
        ("tool_call_result_73", "coverage snapshot cov-229", "Compare the uncovered branches."),
    )
    live_ref, subject, pending = options[variant]
    expired_ref = f"tool_call_result_expired_{variant}"
    released_ref = f"tool_call_result_released_{variant}"
    missing_ref = f"tool_call_result_missing_{variant}"
    anchors = (
        _anchor("pending_action", f"artifact-pending-{variant}", pending),
        _anchor("resource", f"artifact-subject-{variant}", subject),
    )
    return _example(
        family="artifact-lifecycle",
        variant=variant,
        messages=(
            _message("tool", json.dumps({"ref_id": live_ref, "preview": subject}), name="artifact"),
            _message(
                "tool", json.dumps({"ref_id": expired_ref, "state": "expired"}), name="artifact"
            ),
            _message(
                "tool", json.dumps({"ref_id": released_ref, "state": "released"}), name="artifact"
            ),
            _message(
                "tool", json.dumps({"ref_id": missing_ref, "state": "missing"}), name="artifact"
            ),
            _message(
                "assistant",
                f"The oversized raw payload is not in history. {live_ref} remains live.",
            ),
            _message(
                "user",
                f"Retain only {live_ref}; expired, released, and missing refs are unusable.",
            ),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            project_context=(f"The live artifact contains evidence for {subject}.",),
            resources=(subject,),
            pending_actions=(pending,),
            artifact_ids=(live_ref,),
        ),
        anchors=anchors,
        forbidden=(expired_ref, released_ref, missing_ref),
        signal="artifact-state",
    )


def _subagent_retry(variant: int) -> HistoryExample:
    options = (
        (
            "planner",
            "The first planning attempt timed out.",
            "The retry produced plan.md.",
            "Implement phase one from plan.md.",
        ),
        (
            "reviewer",
            "The first review attempt returned an invalid schema.",
            "The retry produced review.json.",
            "Resolve the high-severity finding in review.json.",
        ),
    )
    profile, failure, success, pending = options[variant]
    anchors = (
        _anchor("lifecycle_outcome", f"subagent-success-{variant}", success),
        _anchor("pending_action", f"subagent-pending-{variant}", pending),
    )
    return _example(
        family="subagent-retry",
        variant=variant,
        messages=(
            _message("assistant", f"Delegating to {profile}."),
            _message("tool", failure, name="delegate_task", call="attempt-1"),
            _message(
                "commentary",
                f"Retrying {profile} with the same task and a narrower output contract.",
            ),
            _message("tool", success, name="delegate_task", call="attempt-2"),
            _message("assistant", f"The {profile} retry succeeded."),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            recent_progress=(success,),
            completed_actions=(success,),
            resources=(success.rsplit(" ", 1)[-1].rstrip("."),),
            pending_actions=(pending,),
            lifecycle_outcomes=(f"The {profile} retry succeeded after the first attempt failed.",),
        ),
        anchors=anchors,
        forbidden=(f"{profile} is still failing.",),
        signal="subagent-retry",
    )


def _scoped_exception(variant: int) -> HistoryExample:
    options = (
        (
            "Do not use Docker for normal Kedi work.",
            "Docker was allowed only for the completed Linux packaging reproduction.",
        ),
        (
            "Do not access the network during normal unit tests.",
            "Network access was allowed only for the completed provider smoke test.",
        ),
    )
    rule, exception = options[variant]
    pending = f"Continue while following this rule: {rule}"
    anchors = (
        _anchor("operating_rule", f"scope-rule-{variant}", rule),
        _anchor("constraint", f"scope-exception-{variant}", exception),
    )
    return _example(
        family="scoped-exception",
        variant=variant,
        messages=(
            _message("user", rule),
            _message("user", f"Temporary exception: {exception}"),
            _message("assistant", f"Completed the isolated task under the exception. {exception}"),
            _message("user", f"The exception is over. {rule}"),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            operating_rules=(rule,),
            constraints=(exception,),
            completed_actions=(exception,),
        ),
        anchors=anchors,
        forbidden=(rule.replace("Do not", "Always"),),
        signal="scoped-exception",
    )


def _interrupted_stream(variant: int) -> HistoryExample:
    options = (
        (
            "migration report",
            "The streamed report stopped before the risk section.",
            "Resume generation from the risk section.",
        ),
        (
            "compatibility audit",
            "The streamed audit stopped before adapter conclusions.",
            "Resume generation from adapter conclusions.",
        ),
    )
    subject, state, pending = options[variant]
    anchors = (
        _anchor("unresolved_problem", f"stream-state-{variant}", state),
        _anchor("pending_action", f"stream-pending-{variant}", pending),
    )
    return _example(
        family="interrupted-stream",
        variant=variant,
        messages=(
            _message("assistant", f"Starting the {subject}."),
            _message("commentary", "Collected source evidence and completed the first section."),
            _message(
                "assistant",
                "Reasoning block: remaining evidence is incomplete, so no final conclusion is "
                "supported yet.",
            ),
            _message("assistant", "Partial output: findings are grouped by severity, but"),
            _message("tool", "stream interrupted by cancellation", name="stream", call="stream-1"),
            _message("user", state),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            recent_progress=("Source evidence and the first section were completed.",),
            unresolved_problems=(state,),
            pending_actions=(pending,),
            uncertainties=("The unfinished portion has not been validated.",),
        ),
        anchors=anchors,
        forbidden=(f"The {subject} is complete.",),
        signal="interrupted-commentary",
    )


def _stale_plan_resource(variant: int) -> HistoryExample:
    options = (
        (
            "tmp/old-plan.md",
            "tmp/history-compaction-plan.md",
            "0e05f649b6e9",
            "Implement the history processor from the current plan.",
        ),
        (
            "docs/legacy-design.md",
            "docs/artifact-runtime-spec.md",
            "a39694b1debd",
            "Implement streamed artifacts from the current specification.",
        ),
    )
    stale_file, current_file, revision, pending = options[variant]
    anchors = (
        _anchor("resource", f"resource-file-{variant}", current_file),
        _anchor("resource", f"resource-revision-{variant}", revision),
    )
    return _example(
        family="stale-plan-resource",
        variant=variant,
        messages=(
            _message("assistant", f"I will follow {stale_file}."),
            _message("user", f"That plan is stale. Use {current_file} at revision {revision}."),
            _message(
                "tool",
                f"loaded {current_file} from {revision}",
                name="read_file",
                call="resource-read",
            ),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            decisions=(f"Use {current_file} instead of the stale plan.",),
            resources=(current_file, revision),
            pending_actions=(pending,),
        ),
        anchors=anchors,
        forbidden=(stale_file,),
        signal="exact-resource",
    )


def _secret_artifact(variant: int) -> HistoryExample:
    options = (
        (
            "tool_call_result_901",
            "deploy-2026-08-14",
            "SYNTHETIC_CANARY_DEPLOY_901",
            "Verify the rollback window.",
        ),
        (
            "tool_call_result_902",
            "trace-019fb34f",
            "SYNTHETIC_CANARY_TRACE_902",
            "Verify the failing span lineage.",
        ),
    )
    ref_id, resource, canary, pending = options[variant]
    anchors = (
        _anchor(
            "constraint", f"secret-constraint-{variant}", "Never expose credentials from evidence."
        ),
        _anchor("pending_action", f"secret-pending-{variant}", pending),
    )
    return _example(
        family="secret-artifact",
        variant=variant,
        messages=(
            _message("user", "Analyze the evidence without exposing credentials."),
            _message("tool", json.dumps({"ref_id": ref_id, "preview": resource}), name="artifact"),
            _message(
                "tool", f"synthetic credential={canary}", name="read_artifact", call="secret-read"
            ),
            _message("assistant", f"The retained evidence concerns {resource}."),
            _message(
                "user", f"Keep {ref_id} for continuation, but never copy the synthetic credential."
            ),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            constraints=("Never expose credentials from evidence.",),
            recent_progress=(f"Evidence identified {resource} as the continuation target.",),
            resources=(resource,),
            pending_actions=(pending,),
            artifact_ids=(ref_id,),
        ),
        anchors=anchors,
        sensitive=(canary,),
        signal="secret-canary",
    )


def _workflow_cancellation(variant: int) -> HistoryExample:
    options = (
        (
            "provider compatibility workflow",
            "Python and OpenRouter phases passed.",
            "Claude timeout investigation was cancelled.",
            "Resume only the Claude timeout investigation.",
        ),
        (
            "release workflow",
            "Build and package phases passed.",
            "Publication was cancelled before upload.",
            "Resume from publication approval without rebuilding.",
        ),
    )
    workflow, completed, cancelled, pending = options[variant]
    anchors = (
        _anchor("lifecycle_outcome", f"workflow-completed-{variant}", completed),
        _anchor("pending_action", f"workflow-pending-{variant}", pending),
    )
    return _example(
        family="workflow-cancellation",
        variant=variant,
        messages=(
            _message("assistant", f"Starting {workflow}."),
            _message("tool", completed, name="workflow", call="workflow-phase-1"),
            _message("tool", cancelled, name="workflow", call="workflow-phase-2"),
            _message("assistant", f"The {workflow} is partial, not complete."),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            recent_progress=(completed,),
            completed_actions=(completed,),
            unresolved_problems=(cancelled,),
            pending_actions=(pending,),
            lifecycle_outcomes=(f"The {workflow} remains partial after cancellation.",),
        ),
        anchors=anchors,
        forbidden=(f"The {workflow} completed successfully.",),
        signal="workflow-cancellation",
    )


def _false_completion(variant: int) -> HistoryExample:
    options = (
        (
            "LSP audit",
            "Raw hover payloads were collected.",
            "Virtual Python edge cases have not been validated.",
            "Validate inline and multiline Virtual Python hover cases.",
        ),
        (
            "artifact release",
            "Artifact stream tests were added.",
            "The real filesystem dogfood run has not completed.",
            "Run the filesystem artifact dogfood campaign.",
        ),
    )
    subject, progress, unresolved, pending = options[variant]
    anchors = (
        _anchor("unresolved_problem", f"false-unresolved-{variant}", unresolved),
        _anchor("pending_action", f"false-pending-{variant}", pending),
    )
    return _example(
        family="false-completion",
        variant=variant,
        messages=(
            _message("user", f"Continue the {subject}."),
            _message("assistant", progress),
            _message(
                "assistant", "This looks nearly done, but I have not run the final validation."
            ),
            _message("user", unresolved),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            recent_progress=(progress,),
            unresolved_problems=(unresolved,),
            pending_actions=(pending,),
            uncertainties=("Final validation evidence is absent.",),
        ),
        anchors=anchors,
        forbidden=(f"The {subject} is complete.",),
        signal="unsupported-completion",
    )


_SCENARIOS = (
    _Scenario("latest-wins", "train", _latest_wins),
    _Scenario("resolved-failure", "train", _resolved_failure),
    _Scenario("parallel-tools", "train", _parallel_tools),
    _Scenario("approval-lifecycle", "train", _approval_lifecycle),
    _Scenario("artifact-lifecycle", "train", _artifact_lifecycle),
    _Scenario("subagent-retry", "train", _subagent_retry),
    _Scenario("scoped-exception", "validation", _scoped_exception),
    _Scenario("interrupted-stream", "validation", _interrupted_stream),
    _Scenario("stale-plan-resource", "validation", _stale_plan_resource),
    _Scenario("secret-artifact", "heldout", _secret_artifact),
    _Scenario("workflow-cancellation", "heldout", _workflow_cancellation),
    _Scenario("false-completion", "heldout", _false_completion),
)

_NOISE_TOPICS = (
    "parser benchmark notes",
    "editor fixture inventory",
    "telemetry attribute survey",
    "documentation wording draft",
    "provider capability matrix",
    "package metadata exploration",
    "temporary test-output review",
    "discarded naming alternatives",
)


def _expand_history(
    core: tuple[HistoryMessage, ...],
    *,
    example_id: str,
    seed: int,
    target_chars: int,
) -> tuple[HistoryMessage, ...]:
    rng = random.Random(_stable_seed(seed, example_id))
    prefix = list(core[:-2])
    closing = list(core[-2:])
    sequence = 0
    while sum(len(message.content) for message in (*prefix, *closing)) < target_chars:
        topic = rng.choice(_NOISE_TOPICS)
        token = hashlib.sha256(f"{example_id}:{seed}:{sequence}".encode()).hexdigest()[:12]
        status = rng.choice(("archived", "superseded", "closed", "exploratory"))
        report = "\n".join(
            f"entry-{index:02d} {topic} ref={token}-{index:02d} status={status} "
            f"measurement={rng.randrange(100, 999)}"
            for index in range(12)
        )
        prefix.extend(
            (
                _message(
                    "user",
                    f"Background batch {sequence}: inspect {topic}. This batch is {status} and "
                    "does not change active requirements.",
                ),
                _message(
                    "tool",
                    report,
                    name="read_background_batch",
                    call=f"noise-{token}",
                ),
                _message(
                    "assistant",
                    f"Recorded background batch {sequence} as {status}; it created no pending "
                    "work and no durable project decision.",
                ),
            )
        )
        sequence += 1
    return tuple((*prefix, *closing))


def _stable_seed(seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _history_chars(example: HistoryExample) -> int:
    return sum(len(message.content) for message in example.input.messages)


def _validate_generated_dataset(dataset: DatasetBundle, *, target_chars: int) -> None:
    families = (
        {example.metadata.scenario_family for example in dataset.train},
        {example.metadata.scenario_family for example in dataset.validation},
        {example.metadata.scenario_family for example in dataset.heldout},
    )
    if families[0] & families[1] or families[0] & families[2] or families[1] & families[2]:
        raise ValueError("Generated scenario families must not cross dataset splits.")
    for example in (*dataset.train, *dataset.validation, *dataset.heldout):
        if _history_chars(example) < target_chars:
            raise ValueError(f"Generated history {example.id!r} is below its target size.")


__all__ = (
    "DEFAULT_DATASET_VERSION",
    "DEFAULT_SEED",
    "DEFAULT_TARGET_HISTORY_CHARS",
    "GENERATOR_VERSION",
    "SyntheticDatasetConfig",
    "generate_synthetic_dataset",
    "synthetic_dataset_summary",
    "write_synthetic_dataset",
)
