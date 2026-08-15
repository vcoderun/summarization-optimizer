# Synthetic History Dataset

`synthetic_v2.json` is a deterministic, publication-safe benchmark corpus for Kedi history
summarization. It contains no private conversation data, repository source, production trace, or
real credential. Credential handling is exercised only with conspicuous `SYNTHETIC_CANARY_*`
values that are forbidden from expected and generated checkpoints.

## Provenance

- Dataset version: `synthetic-v2`
- Generator version: `deterministic-kedi-synthetic-v2`
- Seed: `20260815`
- Minimum requested source-history size: `32000` characters per example
- Generator: `kedi_summarization_optimizer.generation`
- Ground truth: deterministic scenario compiler; no model writes expected checkpoints

The checked-in corpus is reproducible with:

```bash
uv run kedi-summarization-optimize generate-dataset datasets/synthetic_v2.json --force
```

The generator does not silently replace an existing dataset unless `--force` is explicit.

## Splits

| Split | Examples | Scenario families | Total history chars | Minimum history chars | SHA-256 fingerprint |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 20 | 10 | 732884 | 33735 | `de6819ace3060d68efd51559bbd27ac0afcf5a4542be0e14861ce8d4adf648e1` |
| Validation | 10 | 5 | 367402 | 34442 | `82365d27d3a0d0e4050f6ed5db88c8a0fa2ece6def8ae68ae0507d4a469f8071` |
| Held-out | 10 | 5 | 379658 | 35343 | `50ea7e9ac19fffa1e3fda107cde86d860e2f5a0f693b57fe70ff1a053df32d12` |

Scenario families never cross split boundaries. The corpus covers Kedi native output and multiline
template semantics, Virtual Python and LSP projection, selective and package imports, skills
resolution, prefix-stable history compaction, approvals, artifacts, subagents, workflows,
interactive execution, adapter parity, telemetry, secret canaries, and unsupported completion.

Long histories are woven from split-local Kedi scenario cores rather than uniform filler labels.
Earlier work items end as completed, superseded, or rejected and carry realistic tool and test
evidence. Active scenario facts appear among those events, while case IDs are opaque and final
messages vary between user, assistant, and commentary roles. Prior-work text does not cross
train/validation/held-out boundaries, apart from two generic continuation tails.

## Ground-truth contract

Each example includes an exact `SummaryCheckpoint`, canonical anchors, forbidden stale claims,
sensitive canaries, valid artifact references, and an output-size boundary. The deterministic
evaluator enforces those hard constraints before a semantic judge can score the output. The
checked-in dataset must remain byte-semantically equal to the default generator output; tests pin
all split fingerprints to prevent code/data drift.

## Publication boundary

This corpus is suitable for a public GitHub or Hugging Face dataset release after an independent
content review. A release should include this data statement, the exact repository revision,
generator settings, split fingerprints, campaign configuration, and redacted Autobench records.
Local `runs/`, authentication state, environment files, and optimizer checkpoints are not dataset
artifacts and must not be published.

The dataset is synthetic evidence for prompt optimization, not proof of production quality. A
release candidate still requires untouched held-out certification and manual review of failure
clusters and model-judge disagreement.
