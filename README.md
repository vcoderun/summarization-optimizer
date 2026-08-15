# Kedi Summarization Optimizer

This repository is the offline optimization and certification pipeline for Kedi's future
stateful-history summarizer. It optimizes one versioned summarizer prompt with Pydantic-GEPA,
records complete optimization and model evidence with Autobench, and publishes the prompt only
after an independent held-out certification campaign passes.

The optimizer is deliberately outside Kedi's production runtime. Kedi will eventually consume an
accepted, versioned prompt artifact; it will not depend on Pydantic-GEPA or Autobench at runtime.

## Current status

The full model-backed pipeline is implemented and has completed an end-to-end pilot:

- GPT-5.6 Luna at high effort writes structured checkpoints.
- GPT-5.6 Terra at high effort judges semantic fidelity.
- GPT-5.6 Terra at high effort reflects on GEPA evidence and proposes improved instructions.
- Pydantic AI and Pydantic-GEPA activity is recorded in the same Autobench experiment.
- The selected candidate is evaluated again on an untouched held-out split.
- Publication is atomic and occurs only when every configured certification gate passes.

The pilot validates the architecture and integrations. Its two-case held-out split is not enough
to claim that the prompt is ready for production Kedi history compaction. A deterministic 40-case,
high-context synthetic corpus now provides the next campaign input, but its results still require
independent review before a prompt can move into Kedi.

## Synthetic dataset

Generate the versioned corpus without model calls:

```bash
uv run kedi-summarization-optimize generate-dataset datasets/synthetic_v2.json
```

Generation is seed-based and deterministic. A scenario compiler owns expected checkpoints,
canonical anchors, stale-state exclusions, lifecycle truth, valid artifact references, and
synthetic secret canaries. Models do not author the ground truth. The twenty Kedi-specific scenario
families are isolated by split, with two variants each and at least 32000 source-history characters
per case. Naturalistic old Kedi tasks replace the v1 corpus's uniform inactive-noise batches.

The checked-in `datasets/synthetic_v2.json` is pinned by split fingerprints and a regression test
that reproduces it from the default generator. See
[datasets/README.md](datasets/README.md) for provenance, coverage, fingerprints, and publication
rules.

## Checkpoint contract

The summarizer reconstructs effective project state rather than producing a transcript summary.
`SummaryCheckpoint` separates:

- the current objective, execution state, unresolved problems, and pending actions;
- durable operating rules, project context, constraints, and active decisions;
- concise recent progress, completed actions, and verified lifecycle outcomes;
- concrete resources, material uncertainties, exact artifact references, and canonical anchors.

Latest applicable instructions win. A later user correction must replace an older conflicting
instruction, resolved failures must not reappear as open problems, and an absent completion claim
must not become an invented failure. Canonical anchors are copied exactly. Artifact IDs may only be
retained when they are explicit opaque artifact references.

Each dataset example also carries a deterministic evaluation contract. Forbidden superseded
phrases and sensitive values may not appear in either the expected checkpoint or model output.
Train, validation, and held-out IDs must be disjoint, and every split receives a stable content
fingerprint.

## Evaluation boundary

Evaluation has two intentionally separate layers.

The deterministic layer rejects outputs that:

- exceed the configured output limit;
- leak forbidden or sensitive text;
- omit, invent, or mutate canonical anchors;
- omit or invent artifact references;
- duplicate entries within a checkpoint section.

Only outputs that pass those gates reach the Terra semantic judge. The judge scores operating-rule,
project-context, current-state, recent-progress, and latest-wins fidelity. A semantic hard pass also
requires grounded output, no critical omissions, a latest-wins score of at least `0.8`, and a total
judge score of at least `0.75`.

Sensitive values are replaced before judge input. A semantic judge cannot override a deterministic
safety or reference-integrity failure.

## Campaign lifecycle

1. Validate the immutable campaign and train/validation/held-out dataset.
2. Run Pydantic-GEPA optimization inside one Autobench experiment.
3. Persist and replay the optimization record before trusting its output.
4. Resolve exactly one selected `summarizer_instructions` asset.
5. Write `selected_candidate.txt`, whether or not the candidate is later accepted.
6. Run a separate Autobench experiment over the untouched held-out split.
7. Persist and replay the certification record.
8. Evaluate mean score, minimum case score, and hard-pass policy.
9. Write `accepted_prompt.txt` only when all publication gates pass.

Optimization and certification records are separate by design. Search-time validation scores never
promote a production prompt on their own.

## Resume semantics

Pydantic-GEPA owns exact optimizer checkpoints under `gepa-checkpoints/`.

- `resume: "never"` requires an empty campaign output directory.
- `resume: "if_exists"` resumes compatible checkpoint state when present.
- `resume: "required"` fails unless checkpoint state exists.
- `fresh: true` starts a new optimizer state and cannot be combined with resume.

Compatibility includes dataset version and fingerprints, import targets, and model configuration.
The pipeline does not warm-start an unrelated campaign from a previous best prompt. A compatible
interrupted campaign resumes exactly; a completed prompt becomes a new baseline only through an
explicitly versioned campaign decision.

Pydantic-GEPA's metric-call budget is a soft boundary between atomic optimizer iterations. A
minibatch that starts below the limit is allowed to finish, so reported usage can exceed
`max_metric_calls` by at most the in-flight atomic work. The pilot configured `24` and completed at
`26` calls.

## Installation and authentication

The lock file pins exact Autobench and Pydantic-GEPA revisions used by the experiment.

```bash
uv sync --all-extras
```

Model-backed campaigns require an authenticated `codex-auth-helper` installation. Credentials are
resolved by that package and must never be placed in campaign files, datasets, records, or committed
environment files.

## Running campaigns

Validate the configuration and immutable split fingerprints without making model calls:

```bash
uv run kedi-summarization-optimize validate examples/pilot_campaign.json
```

Run the model-backed pilot:

```bash
uv run kedi-summarization-optimize run examples/pilot_campaign.json
```

Validate the larger synthetic campaign without spending model calls:

```bash
uv run kedi-summarization-optimize validate examples/synthetic_campaign.json
```

The real synthetic campaign uses Luna high for checkpoint generation and Terra high for reflection
and semantic judging. It has a 96-call GEPA budget, exact checkpoint resume, full Pydantic AI and
Pydantic-GEPA evidence capture, and stricter held-out publication thresholds than the pilot. Run it
only after reviewing the dataset and campaign budget:

```bash
uv run kedi-summarization-optimize run examples/synthetic_campaign.json
```

Replay or report the two durable evidence records independently:

```bash
uv run autobench replay runs/pilot-v1/optimization-record
uv run autobench report runs/pilot-v1/optimization-record
uv run autobench replay runs/pilot-v1/certification-record
uv run autobench report runs/pilot-v1/certification-record
```

A credential-free deterministic smoke campaign remains available for lifecycle regression testing:

```bash
uv run kedi-summarization-optimize run examples/smoke_campaign.json
```

## Campaign outputs

Each campaign writes:

- `gepa-checkpoints/`: exact optimizer-owned resume state;
- `optimization-record/`: immutable Autobench search and model evidence;
- `certification-record/`: immutable held-out evidence;
- `selected_candidate.txt`: the selected candidate, accepted or rejected;
- `accepted_prompt.txt`: created only after successful certification;
- `campaign.json`: the final cross-record decision and reproducibility summary.

Local `runs/` are intentionally ignored. A public experiment release should publish a reviewed,
redacted record bundle and dataset through an explicit release process rather than committing local
runtime state by accident.

## Extension ports

Campaigns select importable `module:callable` targets for invocation, evaluation, and optionally
proposal. Invocation and evaluation targets may be synchronous or asynchronous:

- `invoker_target(instructions, inputs) -> SummaryCheckpoint`
- `evaluator_target(inputs, output, expected) -> CheckpointEvaluation`
- optional `proposer_target(...) -> candidate`

The supplied Codex runtime is one implementation of those ports. Alternative model providers or a
future native Kedi invoker can replace it without changing dataset, evidence, resume, certification,
or publication semantics.

See [docs/PILOT_REPORT.md](docs/PILOT_REPORT.md) for the first real campaign's evidence and limits.
