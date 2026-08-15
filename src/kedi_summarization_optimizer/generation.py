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

GENERATOR_VERSION = "deterministic-kedi-synthetic-v2"
DEFAULT_DATASET_VERSION = "synthetic-v2"
DEFAULT_SEED = 20_260_815
DEFAULT_TARGET_HISTORY_CHARS = 32_000

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


@dataclass(frozen=True)
class _BuiltScenario:
    index: int
    family: str
    split: SplitName
    variant: int
    example: HistoryExample


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
    built = tuple(
        _BuiltScenario(
            index=scenario_index,
            family=scenario.family,
            split=scenario.split,
            variant=variant,
            example=scenario.build(variant),
        )
        for scenario_index, scenario in enumerate(_SCENARIOS, start=1)
        for variant in range(2)
    )
    for item in built:
        base = item.example
        example_id = f"kedi-syn-{item.index:02d}-v{item.variant + 1}"
        prior_work = tuple(
            candidate
            for candidate in built
            if candidate.split == item.split
            and candidate.family != item.family
            and not candidate.example.input.evaluation.sensitive_values
        )
        messages = _expand_history(
            base.input.messages,
            example_id=example_id,
            seed=resolved.seed,
            target_chars=resolved.target_history_chars,
            prior_work=prior_work,
        )
        example = base.model_copy(
            update={
                "id": example_id,
                "input": base.input.model_copy(
                    update={"example_id": example_id, "messages": messages}
                ),
                "metadata": base.metadata.model_copy(
                    update={
                        "labels": {
                            **base.metadata.labels,
                            "seed": str(resolved.seed),
                            "target_history_chars": str(resolved.target_history_chars),
                            "actual_history_chars": str(
                                sum(len(message.content) for message in messages)
                            ),
                            "variant": str(item.variant + 1),
                        }
                    }
                ),
            }
        )
        splits[item.split].append(example)

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
            "Confirm the deployment rollback window from the retained evidence.",
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


def _template_native_output(variant: int) -> HistoryExample:
    options = (
        (
            "Documentation examples should ask the model to return JSON.",
            "Kedi examples must express native language output with [] fields instead of "
            "imperative JSON-return prompts.",
            ">> The capital of <country> is [capital].",
            "docs/language/templates.md now demonstrates blank-filling semantics and its "
            "example validation passed.",
            "Review the remaining examples for imperative output wording.",
        ),
        (
            "Treat each continuation row in a template block as a separate model call.",
            "A >> template block is newline-joined and executed as one model call; a later row "
            "cannot read a field produced by an earlier row in that same block.",
            ">> Select [package_name]\nExplain why <package_name> fits [reason].",
            "The multiline template documentation and parser fixture now agree on one-run "
            "semantics.",
            "Add a regression case for a blank line inside the documented template example.",
        ),
    )
    stale, rule, snippet, progress, pending = options[variant]
    anchors = (
        _anchor("operating_rule", f"template-rule-{variant}", rule),
        _anchor("pending_action", f"template-pending-{variant}", pending),
    )
    return _example(
        family="template-native-output",
        variant=variant,
        messages=(
            _message("user", stale),
            _message("assistant", f"I drafted an example following that approach:\n{snippet}"),
            _message("user", f"That approach is wrong. {rule}"),
            _message("tool", progress, name="docs_validation", call=f"template-docs-{variant}"),
            _message("assistant", progress),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            operating_rules=(rule,),
            project_context=(f"Canonical Kedi example:\n{snippet}",),
            recent_progress=(progress,),
            completed_actions=(progress,),
            resources=("docs/language/templates.md",),
            pending_actions=(pending,),
            lifecycle_outcomes=("The updated documentation example passed validation.",),
        ),
        anchors=anchors,
        forbidden=(stale,),
        signal="native-output-semantics",
    )


def _virtual_python_lsp(variant: int) -> HistoryExample:
    options = (
        (
            "model",
            "str",
            "Inline Python hover mapping returned Unknown for model even though basedpyright "
            "inferred str.",
            "Virtual-document coordinates now map the inline expression back to Kedi and hover "
            "reports str without recoloring the backtick delimiters.",
            "Add multiline Virtual Python hover coverage for an imported function result.",
            "[model: str] = `select_model()`",
        ),
        (
            "solve_knapsack",
            "Callable[[str, int], int]",
            "The imported solve_knapsack symbol had no hover inside a multiline Python block.",
            "Imported Python symbols are now projected into the virtual document and the hover "
            "reports Callable[[str, int], int].",
            "Verify rename and signature help for the imported symbol in a nested procedure.",
            "```\nresult = solve_knapsack(items, capacity)\nreturn result\n```",
        ),
    )
    symbol, inferred_type, failure, fix, pending, snippet = options[variant]
    anchors = (
        _anchor("resource", f"virtual-python-file-{variant}", "src/kedi/lsp/python_virtual.py"),
        _anchor("pending_action", f"virtual-python-pending-{variant}", pending),
    )
    return _example(
        family="virtual-python-lsp",
        variant=variant,
        messages=(
            _message("user", f"This Kedi snippet has a bad hover:\n{snippet}"),
            _message("assistant", failure),
            _message(
                "tool",
                f"basedpyright symbol={symbol} inferred_type={inferred_type} diagnostics=0",
                name="lsp_probe",
                call=f"virtual-probe-{variant}",
            ),
            _message("assistant", fix),
            _message(
                "tool",
                "tests/test_lsp_python_virtual.py: focused hover and semantic-token tests PASSED",
                name="pytest",
                call=f"virtual-tests-{variant}",
            ),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            project_context=(
                "Kedi LSP type information for embedded Python comes from a mapped virtual "
                "Python document.",
            ),
            recent_progress=(fix,),
            completed_actions=(fix,),
            resources=(
                "src/kedi/lsp/python_virtual.py",
                "tests/test_lsp_python_virtual.py",
            ),
            pending_actions=(pending,),
            lifecycle_outcomes=(
                f"The focused Virtual Python tests passed and {symbol} resolves as "
                f"{inferred_type}.",
            ),
        ),
        anchors=anchors,
        forbidden=(failure,),
        signal="virtual-python-type-projection",
    )


def _module_package_import(variant: int) -> HistoryExample:
    options = (
        (
            "Selective imports place only the listed exported names into the environment; Kedi "
            "does not create Python-style aliases or namespace objects.",
            "> import: services/profiles:\n  Profile\n  get_profile",
            "The module fixture confirmed source-order writes: the last declaration or import of "
            "a name wins.",
            "Add a nested relative-import case where two modules export the same value name.",
            ("services/profiles.kedi", "examples/module_imports/main.kedi"),
        ),
        (
            "A package import resolves package/main.kedi, while slash notation selects a "
            "submodule below the declared source tree.",
            "> import: kedi_http/client:\n  request\n  Response",
            "A temporary KEDI_HOME install placed the synthetic package under the Kedi registry "
            "and both root and submodule imports passed.",
            "Verify that a package cannot escape its declared source tree through a symlink.",
            ("package.kedi", "$HOME/.kedi/registry/kedi_http"),
        ),
    )
    rule, snippet, progress, pending, resources = options[variant]
    anchors = (
        _anchor("decision", f"import-rule-{variant}", rule),
        _anchor("pending_action", f"import-pending-{variant}", pending),
    )
    return _example(
        family="module-package-import",
        variant=variant,
        messages=(
            _message("user", f"Use this Kedi import shape in the fixture:\n{snippet}"),
            _message("assistant", "I initially modeled it as a Python module namespace."),
            _message("user", f"Do not apply Python import assumptions. {rule}"),
            _message("tool", progress, name="kedi_parse_and_run", call=f"import-run-{variant}"),
            _message("assistant", progress),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            project_context=(rule,),
            decisions=(rule,),
            recent_progress=(progress,),
            completed_actions=(progress,),
            resources=resources,
            pending_actions=(pending,),
        ),
        anchors=anchors,
        forbidden=("Create a Python-style module alias.",),
        signal="kedi-module-resolution",
    )


def _skills_registry_resolution(variant: int) -> HistoryExample:
    options = (
        (
            "Use > skills:, not > use: skills. Resolve the Kedi registry first, then local "
            ".agents/skills, then the user's global .agents/skills directory.",
            "> skills:\n  enabled: true\n  include_registry: true\n  include_all: false",
            "The resolver selected the registry copy of skill-creator and list_skills returned "
            "stable IDs without reading SKILL.md eagerly.",
            "Add exclude_paths coverage when the same skill exists in all three roots.",
        ),
        (
            "Enabled skills are discovered explicitly; list_skills returns IDs and read_skill "
            "loads one SKILL.md only when the agent chooses it.",
            "> skills:\n  enabled: true\n  max_skills: 12\n"
            '  exclude_paths: `["~/.agents/skills/legacy"]`',
            "The Kedi skills fixture loaded one requested skill and left the remaining skill "
            "contents outside model context.",
            "Kedi registry ile local skill ayni ada sahipse precedence testini ekle.",
        ),
    )
    rule, snippet, progress, pending = options[variant]
    anchors = (
        _anchor("operating_rule", f"skills-rule-{variant}", rule),
        _anchor("pending_action", f"skills-pending-{variant}", pending),
    )
    return _example(
        family="skills-registry-resolution",
        variant=variant,
        messages=(
            _message("user", f"Configure the profile with this syntax:\n{snippet}"),
            _message("assistant", "I was going to register skills through > use: skills."),
            _message("user", rule),
            _message("tool", progress, name="skill_smoke", call=f"skills-smoke-{variant}"),
            _message("assistant", progress),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            operating_rules=(rule,),
            project_context=("A Kedi skill is a one-file SKILL.md capability read explicitly.",),
            recent_progress=(progress,),
            completed_actions=(progress,),
            resources=("$HOME/.kedi/registry/skills", "./.agents/skills"),
            pending_actions=(pending,),
        ),
        anchors=anchors,
        forbidden=("Register skills through > use: skills.",),
        signal="skill-discovery-precedence",
    )


def _history_prefix_compaction(variant: int) -> HistoryExample:
    options = (
        (
            "Preserve the existing message order and append new turns; do not reorder a cached "
            "prefix merely because recent files were accessed in a different order.",
            "The history processor retained the cached prefix byte-for-byte and removed only "
            "eligible suffix noise.",
            "Add a cache regression where B reads d, c, b, x after A cached a, b, c, d.",
            "tests/test_history_processor.py",
        ),
        (
            "Releasing an artifact removes it from the live store but must not delete or rewrite "
            "earlier history messages that are part of a provider-cached prefix.",
            "The release path now marks the reference unavailable while preserving append-only "
            "conversation order.",
            "Verify compaction turns the released reference into a compact lifecycle fact without "
            "changing the cached prefix.",
            "tests/test_native_compaction.py",
        ),
    )
    rule, progress, pending, test_file = options[variant]
    anchors = (
        _anchor("operating_rule", f"history-rule-{variant}", rule),
        _anchor("pending_action", f"history-pending-{variant}", pending),
    )
    return _example(
        family="history-prefix-compaction",
        variant=variant,
        messages=(
            _message("user", rule),
            _message("assistant", "I considered rebuilding history in most-recently-used order."),
            _message("user", "That would destroy the provider cache prefix. Keep append order."),
            _message("tool", progress, name="history_probe", call=f"history-probe-{variant}"),
            _message("assistant", progress),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            operating_rules=(rule,),
            project_context=(
                "Kedi history compaction must preserve unchanged provider-cached prefixes.",
            ),
            recent_progress=(progress,),
            completed_actions=(progress,),
            resources=(test_file,),
            pending_actions=(pending,),
        ),
        anchors=anchors,
        forbidden=("Rebuild history in most-recently-used order.",),
        signal="cache-stable-history",
    )


def _telemetry_adapter_parity(variant: int) -> HistoryExample:
    options = (
        (
            "Use service.name=kedi. Instrumentation scope names may be kedi.<scope>, while visible "
            "span names remain human-readable actions such as agent reviewer and call read_file.",
            "The fake Pydantic and LangChain runs emitted one agent span and one tool span each "
            "without duplicate Pydantic AI spans.",
            "Run the same parity probe for the Codex adapter and compare semantic attributes.",
        ),
        (
            "Kedi telemetry is a no-op until a backend is installed; adapter shims add agent and "
            "tool spans only where the underlying framework does not already provide them.",
            "The no-backend runtime produced no exports, and the installed test backend captured "
            "subagent, workflow, approval, and artifact lifecycle spans.",
            "Add cancellation status assertions for a dynamic workflow span.",
        ),
    )
    rule, outcome, pending = options[variant]
    anchors = (
        _anchor("operating_rule", f"telemetry-rule-{variant}", rule),
        _anchor("pending_action", f"telemetry-pending-{variant}", pending),
    )
    return _example(
        family="telemetry-adapter-parity",
        variant=variant,
        messages=(
            _message("user", rule),
            _message("assistant", "I added generic kedi.agent.invoke and kedi.tool.execute spans."),
            _message(
                "user",
                "Those names are not the agreed DX. Keep semantic actions run_agent/call_tool, "
                "but make visible names readable.",
            ),
            _message("tool", outcome, name="otel_test_backend", call=f"otel-probe-{variant}"),
            _message("assistant", outcome),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            operating_rules=(rule,),
            recent_progress=(outcome,),
            completed_actions=(outcome,),
            resources=("kedi-otel", "src/kedi/telemetry"),
            pending_actions=(pending,),
            lifecycle_outcomes=("The focused telemetry parity probe passed.",),
        ),
        anchors=anchors,
        forbidden=("kedi.agent.invoke", "kedi.tool.execute"),
        signal="otel-adapter-parity",
    )


def _interactive_session_state(variant: int) -> HistoryExample:
    options = (
        (
            "InteractiveSession.execute_fragment executes only the new fragment and preserves "
            "the existing Kedi environment without rerunning earlier side effects.",
            "Two fragments shared x=4; the side-effect counter remained one and :show x "
            "returned 4.",
            "Add a fragment that imports a selective Kedi module after an earlier assignment.",
        ),
        (
            "The terminal REPL uses +++ as its configurable begin marker, ... for continuation, "
            "and :show <expr> for explicit expression display.",
            "A multiline procedure accepted continuation input and Ctrl-C exited without printing "
            "a Python KeyboardInterrupt traceback.",
            "REPL importundan sonra onceki fragmentin tekrar calismadigini test et.",
        ),
    )
    rule, outcome, pending = options[variant]
    anchors = (
        _anchor("project_fact", f"interactive-rule-{variant}", rule),
        _anchor("pending_action", f"interactive-pending-{variant}", pending),
    )
    return _example(
        family="interactive-session-state",
        variant=variant,
        messages=(
            _message("user", rule),
            _message("assistant", "I first implemented each fragment by calling run_main again."),
            _message(
                "user", "Do not change run_main; incremental execution is a separate surface."
            ),
            _message("tool", outcome, name="interactive_smoke", call=f"idle-smoke-{variant}"),
            _message("assistant", outcome),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            project_context=(rule,),
            constraints=("Do not change run_main; incremental execution is a separate surface.",),
            recent_progress=(outcome,),
            completed_actions=(outcome,),
            resources=("src/kedi/interactive_session.py", "src/kedi/idle.py"),
            pending_actions=(pending,),
        ),
        anchors=anchors,
        forbidden=("Implement each fragment by calling run_main again.",),
        signal="incremental-execution-state",
    )


def _adapter_profile_parity(variant: int) -> HistoryExample:
    options = (
        (
            "Native PydanticAdapter run, iter, and run_stream calls must apply the active Kedi "
            "profile's system prompt, model settings, effort, MCP servers, scoped tools, and "
            "required-tool policy.",
            "The parity matrix passed for run and iter, including external approval deferral with "
            "canonical Kedi arguments.",
            "Add early-close cancellation coverage for run_stream.",
        ),
        (
            "LazyAdapter must resolve and forward child execution, approval, history, "
            "stream-event, and model-conversion capabilities instead of hiding the concrete "
            "adapter surface.",
            "A Pydantic-backed LazyAdapter delegated to planner and returned task_summary plus "
            "final_result without the previous unsupported-delegation error.",
            "Verify the same capability forwarding after a profile override changes adapters.",
        ),
    )
    rule, outcome, pending = options[variant]
    stale = "The active adapter does not support Kedi subagent delegation."
    anchors = (
        _anchor("operating_rule", f"adapter-rule-{variant}", rule),
        _anchor("pending_action", f"adapter-pending-{variant}", pending),
    )
    return _example(
        family="adapter-profile-parity",
        variant=variant,
        messages=(
            _message("user", rule),
            _message("tool", stale, name="delegate_task", call=f"adapter-fail-{variant}"),
            _message("assistant", "The wrapper exposed only the base invoke method."),
            _message("assistant", outcome),
            _message("tool", "focused adapter parity tests PASSED", name="pytest"),
            _message("user", pending),
        ),
        expected=SummaryCheckpoint(
            current_objective=pending,
            operating_rules=(rule,),
            recent_progress=(outcome,),
            completed_actions=(outcome,),
            resources=("src/kedi/agent_adapter", "tests/test_pydantic_native_parity.py"),
            pending_actions=(pending,),
            lifecycle_outcomes=("The focused adapter parity tests passed after the fix.",),
        ),
        anchors=anchors,
        forbidden=(stale,),
        signal="adapter-capability-forwarding",
    )


_SCENARIOS = (
    _Scenario("latest-wins", "train", _latest_wins),
    _Scenario("resolved-failure", "train", _resolved_failure),
    _Scenario("parallel-tools", "train", _parallel_tools),
    _Scenario("approval-lifecycle", "train", _approval_lifecycle),
    _Scenario("artifact-lifecycle", "train", _artifact_lifecycle),
    _Scenario("subagent-retry", "train", _subagent_retry),
    _Scenario("template-native-output", "train", _template_native_output),
    _Scenario("virtual-python-lsp", "train", _virtual_python_lsp),
    _Scenario("skills-registry-resolution", "train", _skills_registry_resolution),
    _Scenario("history-prefix-compaction", "train", _history_prefix_compaction),
    _Scenario("scoped-exception", "validation", _scoped_exception),
    _Scenario("interrupted-stream", "validation", _interrupted_stream),
    _Scenario("stale-plan-resource", "validation", _stale_plan_resource),
    _Scenario("module-package-import", "validation", _module_package_import),
    _Scenario("interactive-session-state", "validation", _interactive_session_state),
    _Scenario("secret-artifact", "heldout", _secret_artifact),
    _Scenario("workflow-cancellation", "heldout", _workflow_cancellation),
    _Scenario("false-completion", "heldout", _false_completion),
    _Scenario("telemetry-adapter-parity", "heldout", _telemetry_adapter_parity),
    _Scenario("adapter-profile-parity", "heldout", _adapter_profile_parity),
)


def _expand_history(
    core: tuple[HistoryMessage, ...],
    *,
    example_id: str,
    seed: int,
    target_chars: int,
    prior_work: tuple[_BuiltScenario, ...],
) -> tuple[HistoryMessage, ...]:
    rng = random.Random(_stable_seed(seed, example_id))
    target = target_chars + rng.randrange(max(1, target_chars // 5))
    work_items = list(prior_work)
    rng.shuffle(work_items)
    if not work_items:
        raise ValueError("Synthetic history expansion requires split-local prior work.")

    history = [core[0]]
    next_core = 1
    sequence = 0
    while sum(len(message.content) for message in history) < target:
        if next_core < len(core) - 1 and (sequence == 0 or rng.random() < 0.34):
            history.append(core[next_core])
            next_core += 1
        prior = work_items[sequence % len(work_items)]
        token = hashlib.sha256(f"{example_id}:{seed}:{sequence}".encode()).hexdigest()[:12]
        history.extend(
            _resolved_prior_work(
                prior,
                token=token,
                sequence=sequence,
                rng=rng,
            )
        )
        sequence += 1

    history.extend(core[next_core:-1])
    history.append(core[-1])
    tail_mode = _stable_seed(seed + 1, example_id) % 3
    if tail_mode == 1:
        history.append(
            _message(
                "commentary",
                "I am opening the referenced Kedi source and focused tests for this request now.",
            )
        )
    elif tail_mode == 2:
        history.append(_message("assistant", "I will continue with that exact scope."))
    return tuple(history)


def _resolved_prior_work(
    prior: _BuiltScenario,
    *,
    token: str,
    sequence: int,
    rng: random.Random,
) -> tuple[HistoryMessage, ...]:
    work_item = f"KEDI-SYN-{token.upper()}"
    copied: list[HistoryMessage] = [
        _message(
            "commentary",
            f"Earlier work item {work_item} concerned {prior.family.replace('-', ' ')}.",
        )
    ]
    sensitive_values = prior.example.input.evaluation.sensitive_values
    for message in prior.example.input.messages:
        content = message.content
        for value in sensitive_values:
            content = content.replace(value, "<REDACTED_SYNTHETIC_CANARY>")
        copied.append(
            _message(
                message.role,
                content,
                name=message.name,
                call=(
                    f"{message.tool_call_id}-{token}" if message.tool_call_id is not None else None
                ),
            )
        )

    states = ("completed", "superseded", "rejected")
    state = states[sequence % len(states)]
    report = _prior_work_report(
        family=prior.family,
        work_item=work_item,
        state=state,
        rng=rng,
    )
    copied.append(
        _message(
            "tool",
            report,
            name="prior_work_resolution",
            call=f"prior-resolution-{token}",
        )
    )
    closure = {
        "completed": (
            f"{work_item} belonged to an earlier milestone. Its focused checks passed and no "
            "follow-up from that work item remains active."
        ),
        "superseded": (
            f"{work_item} was replaced before the current task. Do not carry its requested "
            "changes or constraints forward."
        ),
        "rejected": (
            f"{work_item} was not accepted and made no source change. Do not resume its proposal."
        ),
    }[state]
    copied.append(_message("user", closure))
    return tuple(copied)


def _prior_work_report(
    *,
    family: str,
    work_item: str,
    state: str,
    rng: random.Random,
) -> str:
    test_module = family.replace("-", "_")
    rows = [f"work_item={work_item} family={family} terminal_state={state}"]
    rows.extend(
        f"tests/test_{test_module}.py::test_case_{index:02d} "
        f"PASSED duration_ms={rng.randrange(8, 900)} "
        f"trace={rng.randrange(100000, 999999)}"
        for index in range(32)
    )
    return "\n".join(rows)


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
