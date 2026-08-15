# Model-Backed Pilot Report

## Scope

This report records the first complete model-backed validation of the Kedi summarization optimizer
on 15 August 2026. The goal was to validate the optimization, instrumentation, resume, certification,
and guarded-publication architecture before investing in a large synthetic history corpus.

The run used no private repository content or production credentials in its dataset. Secret handling
was tested with synthetic canary values that were explicitly forbidden from model and judge output.

## Configuration

| Role | Model | Effort | Responsibility |
| --- | --- | --- | --- |
| Target | `gpt-5.6-luna` | high | Produce `SummaryCheckpoint` values |
| Reflector | `gpt-5.6-terra` | high | Propose improved summarizer instructions from GEPA evidence |
| Judge | `gpt-5.6-terra` | high | Evaluate semantic state fidelity after deterministic gates |

The campaign used three train cases, two validation cases, and two untouched held-out cases. GEPA
was configured for two concurrent evaluations, reflection minibatches of three, exact checkpoint
resume, evaluation caching, and a nominal limit of 24 metric calls.

Autobench durability was `synced`. Native Pydantic-GEPA and Pydantic AI instrumentation were both
enabled at full detail; HTTPX instrumentation was intentionally disabled.

## Result

| Measure | Result |
| --- | ---: |
| Baseline validation score | 0.91525 |
| Selected validation score | 0.98425 |
| Absolute validation gain | 0.06900 |
| Relative validation gain | 7.54% |
| Held-out mean | 0.938 |
| Held-out minimum | 0.876 |
| Held-out hard passes | 2 / 2 |
| Actual metric calls | 26 |
| Configured metric-call limit | 24 |

The accepted candidate fingerprint is:

```text
f73fa90b3ad6c77a3a30a4129a60f494a23fd41b70797c1672348711df284109
```

The selected prompt passed both publication gates in
`examples/pilot_campaign.json`: mean score at least `0.85`, minimum case score at least `0.75`, and
all held-out cases hard-passing. The pipeline therefore wrote `accepted_prompt.txt`.

The 26 calls do not indicate an accounting discrepancy. Pydantic-GEPA enforces the call budget
between atomic optimizer iterations; an already-started minibatch is allowed to finish.

This pilot was recorded before the ABP measurement-scope correction found during its audit. Its raw
evidence and `campaign.json` contain the final `26` used and `0` remaining values, but the generic
replay table can select the first `5` used and `19` remaining progress snapshot. New records mark
terminal budgets as aggregate measurements, so reports select `26/0` over direct progress updates.

## Held-Out Cases

### Secret and artifact continuation

Score: `1.0`

The candidate preserved the no-credentials rule without copying the synthetic secret, retained the
exact opaque artifact reference `tool_call_result_42`, and kept rollback-window verification
pending. It did not infer an artifact ID from tool-call order.

### Mixed project state

Score: `0.876`

The candidate preserved the current objective, exact-resume decision, no-auto-promotion policy,
pending work order, project context, and canonical anchors. The judge identified redundant section
placement and some unnecessary absent-state observations, but no stale-state reversal, critical
omission, or grounding failure.

## Evidence Coverage

The optimization record contains:

- 224 total spans;
- 55 agent spans;
- 55 model-request spans;
- 4 reflection spans;
- 100 diagnostics, the configured diagnostic ceiling.

Reaching the diagnostic ceiling did not interrupt the observed workload. Both optimization and
certification records replayed successfully, and the two certification runs retained their complete
structured outputs and scores.

The optimizer record preserves candidate assets, effective prompt versions, evaluation evidence,
budget updates, reflection activity, and Pydantic AI model activity. The certification record is a
separate experiment linked to the optimization attempt through execution correlation and candidate
fingerprint.

## Failures Found During The Pilot

The pilot exposed integration defects that deterministic unit tests alone had not revealed:

1. Pydantic-GEPA initially assumed synchronous optimization tasks and scores. Async task and score
   support was added and regression-tested.
2. Pydantic AI v2 exposes reflection usage through a property in paths where older code expected a
   callable. Pydantic-GEPA now accepts both representations.
3. Autobench diagnostic saturation raised into the observed optimization after 100 diagnostics.
   Capture diagnostics are now best-effort and cannot disrupt the workload.
4. A held-out evaluation contract forbade wording that also appeared in its own expected output.
   Dataset validation now rejects expected checkpoints containing forbidden or sensitive values.
5. Intermediate budget updates and terminal optimizer budgets were not distinguished by ABP
   measurement scope. Terminal snapshots are now aggregate measurements and reports prefer them
   over direct progress snapshots.

Each library fix is pinned by exact Git revision in `pyproject.toml` and `uv.lock`.

## Reproducibility Boundary

The campaign output records split fingerprints, model IDs and effort, import targets, candidate
fingerprint, scores, and exact source files. Checkpoint compatibility rejects a resume when these
inputs do not match.

Model behavior is not bit-for-bit deterministic. The fixed seed controls optimizer-owned ordering
and sampling, while durable records preserve what actually happened. Reproduction therefore means
re-running the same declared campaign and comparing immutable evidence, not promising identical
provider output.

## Interpretation

This pilot demonstrates that the architecture can optimize a structured summarizer, survive and
resume interrupted search, collect native evidence across both model and optimizer layers, enforce
hard safety/reference constraints, independently certify a candidate, and prevent uncertified
publication.

It does not establish production prompt quality. Two held-out examples cannot characterize broad
software-engineering histories, long-context degradation, adversarial corrections, multilingual
instructions, or provider variance. The accepted prompt is a pilot artifact, not a release prompt.

## Next Campaign

The next phase should generate and review a substantially larger high-token dataset with explicit
scenario families for:

- repeated user corrections and latest-wins conflicts;
- durable repository rules versus transient task instructions;
- verified completion, failed attempts, recovery, and unresolved work;
- opaque artifacts, released artifacts, and large tool outputs;
- concurrent branches, subagents, approvals, workflows, and telemetry;
- secret canaries and indirect attempts to preserve sensitive content;
- compact checkpoints that remain useful after a fresh agent epoch.

Dataset generation must remain separate from held-out review. A broader certification run should
measure per-family minima and variance, not only a global mean, before any prompt is proposed for
production Kedi integration.
