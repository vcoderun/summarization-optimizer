# Synthetic History Dataset

`synthetic_v1.json` is a deterministic, publication-safe benchmark corpus for Kedi history
summarization. It contains no private conversation data, repository source, production trace, or
real credential. Credential handling is exercised only with conspicuous `SYNTHETIC_CANARY_*`
values that are forbidden from expected and generated checkpoints.

## Provenance

- Dataset version: `synthetic-v1`
- Generator version: `deterministic-synthetic-v1`
- Seed: `20260815`
- Minimum requested source-history size: `24000` characters per example
- Generator: `kedi_summarization_optimizer.generation`
- Ground truth: deterministic scenario compiler; no model writes expected checkpoints

The checked-in corpus is reproducible with:

```bash
uv run kedi-summarization-optimize generate-dataset datasets/synthetic_v1.json --force
```

The generator does not silently replace an existing dataset unless `--force` is explicit.

## Splits

| Split | Examples | Scenario families | Total history chars | Minimum history chars | SHA-256 fingerprint |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 12 | 6 | 298191 | 24335 | `baea88d0903de585fd24a2b5316afcdb0cf426c9dc58342712510b29dd8e3a8b` |
| Validation | 6 | 3 | 148588 | 24688 | `691fdef26936bcf3a81d4df97d6062e85a0452491ec33dbc1f4f0bab329ba6ba` |
| Held-out | 6 | 3 | 148254 | 24541 | `acd3e03328c4ec78251bc39ec1075ecc520541e1ea3736e2ded59a09c3d3c40d` |

Scenario families never cross split boundaries. Train covers latest-wins corrections, resolved
failures, parallel tools, approvals, artifact lifecycle, and subagent retry. Validation covers
scoped exceptions, interrupted streams, and stale resources. Held-out certification covers secret
canaries, workflow cancellation, and unsupported completion claims.

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
