"""Typed campaign configuration and path normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


class CodexModelRole(FrozenConfig):
    model_id: str = Field(min_length=1)
    effort: ReasoningEffort = "high"


class CodexModelsSettings(FrozenConfig):
    summarizer: CodexModelRole = Field(
        default_factory=lambda: CodexModelRole(model_id="gpt-5.6-luna")
    )
    reflector: CodexModelRole = Field(
        default_factory=lambda: CodexModelRole(model_id="gpt-5.6-terra")
    )
    judge: CodexModelRole = Field(default_factory=lambda: CodexModelRole(model_id="gpt-5.6-terra"))


class GEPASettings(FrozenConfig):
    max_metric_calls: int = Field(default=100, ge=1)
    max_concurrency: int = Field(default=5, ge=1)
    reflection_model: str | None = None
    reflection_target: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$",
    )
    proposer_target: str | None = None
    reflection_minibatch_size: int = Field(default=3, ge=1)
    seed: int = 0
    resume: Literal["never", "if_exists", "required"] = "never"
    fresh: bool = False
    checkpoint_interval: int = Field(default=1, ge=1)
    cache_evaluations: bool = True

    @model_validator(mode="after")
    def require_reflection_strategy(self) -> GEPASettings:
        choices = (
            self.reflection_model is not None,
            self.reflection_target is not None,
            self.proposer_target is not None,
        )
        if sum(choices) != 1:
            raise ValueError(
                "Set exactly one of reflection_model, reflection_target, or proposer_target."
            )
        if self.fresh and self.resume != "never":
            raise ValueError("fresh cannot be combined with a resume mode.")
        return self


class InstrumentationSettings(FrozenConfig):
    detail: Literal["summary", "evaluations", "full"] = "evaluations"
    include_pydantic_ai: bool = True
    include_httpx: bool = False


class CertificationSettings(FrozenConfig):
    minimum_mean_score: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_case_score: float = Field(default=0.75, ge=0.0, le=1.0)
    require_all_hard_pass: bool = True


class CampaignConfig(FrozenConfig):
    campaign_id: str = Field(min_length=1)
    seed_instructions: str = Field(min_length=1)
    dataset_path: Path
    output_dir: Path
    invoker_target: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
    evaluator_target: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
    models: CodexModelsSettings = Field(default_factory=CodexModelsSettings)
    gepa: GEPASettings
    instrumentation: InstrumentationSettings = Field(default_factory=InstrumentationSettings)
    certification: CertificationSettings = Field(default_factory=CertificationSettings)
    record_durability: Literal["atomic", "synced"] = "atomic"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "gepa-checkpoints"

    @property
    def optimization_record_dir(self) -> Path:
        return self.output_dir / "optimization-record"

    @property
    def certification_record_dir(self) -> Path:
        return self.output_dir / "certification-record"


def load_campaign_config(path: str | Path) -> CampaignConfig:
    source = Path(path).expanduser().resolve()
    config = CampaignConfig.model_validate_json(source.read_text(encoding="utf-8"))
    base = source.parent
    return config.model_copy(
        update={
            "dataset_path": _resolve_path(config.dataset_path, base),
            "output_dir": _resolve_path(config.output_dir, base),
        }
    )


def _resolve_path(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    return (base / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()


__all__ = (
    "CampaignConfig",
    "CertificationSettings",
    "CodexModelRole",
    "CodexModelsSettings",
    "GEPASettings",
    "InstrumentationSettings",
    "load_campaign_config",
)
