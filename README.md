# Kedi Summarization Optimizer

This is the offline optimization and certification pipeline for Kedi's future
history summarizer. It deliberately does not implement model invocation or
synthetic dataset generation yet. Those concerns attach through importable
ports and the versioned dataset contract.

## Current lifecycle

1. Validate the campaign config and immutable train/validation/held-out dataset.
2. Run a real pydantic-gepa optimization inside one Autobench experiment.
3. Record and replay native optimizer evidence through
   `PydanticGEPAInstrumentation`.
4. Resolve exactly one selected `summarizer_instructions` asset.
5. Run a separate Autobench experiment on the untouched held-out split.
6. Publish `accepted_prompt.txt` only if every configured certification gate
   passes.

The optimizer record and certification record remain separate. GEPA search
scores cannot promote a production prompt by themselves.

## Extension ports

Campaigns provide three `module:callable` targets:

- `invoker_target(instructions, inputs) -> SummaryCheckpoint`
- `evaluator_target(inputs, output, expected) -> CheckpointEvaluation`
- `proposer_target(...) -> candidate` or a pydantic-gepa reflection model

The future Kedi invoker and synthetic corpus generator can replace the supplied
deterministic smoke fixtures without changing the pipeline lifecycle.

## Smoke run

```bash
uv sync --extra dev
uv run kedi-summarization-optimize validate examples/smoke_campaign.json
uv run kedi-summarization-optimize run examples/smoke_campaign.json
uv run autobench replay runs/smoke/optimization-record
uv run autobench report runs/smoke/optimization-record
uv run autobench replay runs/smoke/certification-record
```

The smoke adapter performs no network or model I/O. It exercises actual
pydantic-gepa optimization, native Autobench instrumentation, immutable records,
replay, held-out certification, and guarded prompt publication.

Both Pydantic-GEPA and Pydantic AI instrumentation are enabled by default. The
deterministic smoke invoker does not call Pydantic AI, so its record contains
GEPA evidence but no Pydantic AI spans. A future Pydantic AI-backed invoker will
emit model and tool activity into the same Autobench run automatically.

## Output boundary

Each campaign writes:

- `gepa-checkpoints/`: optimizer-owned resume state;
- `optimization-record/`: immutable Autobench search evidence;
- `certification-record/`: immutable held-out evidence;
- `selected_candidate.txt`: the optimizer selection, accepted or not;
- `accepted_prompt.txt`: present only after certification succeeds;
- `campaign.json`: the final cross-record decision summary.

Production Kedi must consume only an accepted, versioned prompt artifact and
its reproducibility metadata. It must not depend on Autobench or pydantic-gepa.
