"""Importable extension ports for invocation, evaluation, and proposal."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from importlib import import_module
from typing import Protocol, TypeVar, cast

from pydantic_gepa.harness import run_awaitable_sync
from pydantic_gepa.values import SerializableValue

from .models import CheckpointEvaluation, SummarizationInput, SummaryCheckpoint

T = TypeVar("T")


class SummarizerInvoker(Protocol):
    def __call__(
        self,
        instructions: str,
        inputs: SummarizationInput,
    ) -> SummaryCheckpoint | Awaitable[SummaryCheckpoint]: ...


class CheckpointEvaluator(Protocol):
    def __call__(
        self,
        inputs: SummarizationInput,
        output: SummaryCheckpoint,
        expected: SummaryCheckpoint,
    ) -> CheckpointEvaluation | Awaitable[CheckpointEvaluation]: ...


class CandidateProposer(Protocol):
    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, SerializableValue]]],
        components_to_update: list[str],
    ) -> dict[str, str]: ...


def resolve_target(target: str) -> Callable[..., object]:
    module_name, separator, attribute_path = target.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError("Import targets must use 'module:attribute' format.")
    value: object = import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"Import target is not callable: {target}")
    return cast("Callable[..., object]", value)


def invoke(
    target: SummarizerInvoker,
    instructions: str,
    inputs: SummarizationInput,
) -> SummaryCheckpoint:
    result = target(instructions, inputs)
    if inspect.isawaitable(result):
        result = run_awaitable_sync(cast("Awaitable[SummaryCheckpoint]", result))
    return SummaryCheckpoint.model_validate(result)


def evaluate(
    target: CheckpointEvaluator,
    inputs: SummarizationInput,
    output: SummaryCheckpoint,
    expected: SummaryCheckpoint,
) -> CheckpointEvaluation:
    result = target(inputs, output, expected)
    if inspect.isawaitable(result):
        result = run_awaitable_sync(cast("Awaitable[CheckpointEvaluation]", result))
    return CheckpointEvaluation.model_validate(result)


def typed_target(target: str, protocol: type[T]) -> T:
    del protocol
    return cast("T", resolve_target(target))


__all__ = (
    "CandidateProposer",
    "CheckpointEvaluator",
    "SummarizerInvoker",
    "evaluate",
    "invoke",
    "resolve_target",
    "typed_target",
)
