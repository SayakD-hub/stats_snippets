# Agent debugging & fix instructions

Working notes for Cursor / coding agents in `ILI_workflow_dev`. Prefer these over clever architecture when the user has already named the bug.

## Core rule: fix the bug that was stated

When the user describes a concrete failure mode, **solve that failure mode first**. Do not start by:

- renaming APIs for consistency
- inventing a more “pure” data model
- copying patterns from another workflow “because RCNN does X”
- extracting helpers that encode dual semantics

If the stated bug is “reads use the write bucket,” the fix is “stop using the write bucket for reads” — usually an optional write param or a one-line caller change — not a new input-bucket concept, stamped segment fields, or URI resolvers.

### Checklist before proposing a design

1. **Restate the bug in one sentence** the user already used (or would recognize).
2. **Point at the exact coupling** (which param / which call site uses the wrong value).
3. **Propose the smallest change** that breaks that coupling.
4. **Only then** ask whether naming, DRY, or shared helpers are worth a follow-up.

If step 3 is enough and the user asked for minimal, stop there.

## Prefer the callee’s call site over the callee’s current semantics

Wrong approach (what went wrong on the S3 grouping fix):

> `s3_bucket` currently means “write” inside `run_grouping_pipeline`, so add `s3_input_bucket` and keep write as the default meaning.

Right approach (what the user asked for first):

> Discovery already passes the read bucket; callers should pass that as `s3_bucket` for reads and an optional `s3_output_bucket` for writes.

Let the **caller’s intent** (read vs write) drive the API. Do not treat the current overloaded meaning of a parameter as sacred if the overload *is* the bug.

## Do not expand optional dual-location into architecture

For split inference / grouping S3 locations:

| Do | Don’t |
|----|--------|
| Optional write override (`s3_output_bucket` or equivalent) | Flip default meaning of `s3_bucket` without a clear plan |
| Keep discovery on inference location | Stamp `bucket` on every segment unless many callers need self-describing URIs |
| Reuse existing helpers (`effective_s3_grouping_output_location`) | Re-implement `getattr` / if-else that the helper already owns |
| DRY path wiring only after the kwargs are correct | Put dual-bucket semantics inside a big resolve helper as the “fix” |

A shared `_resolve_grouping_io` (or similar) is fine **after** the kwargs are `s3_bucket` (read) + `s3_output_bucket` (write). It must not invent a new bucket concept.

## Scope discipline

- **Do not touch** sibling workflows (e.g. OD corrosion) unless the user asks or they share the broken call site.
- **Do not merge** unrelated cleanups (length/odo alignment, filename schemes, stamp-URI refactors) into the same PR as a one-line I/O bug unless the user explicitly expands scope.
- **Tests:** only keep assertions that must change for the new API (e.g. write bucket from `s3_output_bucket`). Drop piggybacked cross-bucket tests on unrelated cases and duplicate “also assert read bucket from pipeline kwargs” checks when discovery already asserts read location.

## When a reviewer suggests two options

Pick the option that matches the **stated bug and minimal surface**, not the option that best matches another codebase’s long-term shape — unless the user asks for that migration.

Example: “stamp full URI like RCNN” vs “optional `s3_output_bucket`” — for corrosion’s key-only segments and a write-bucket bug, the optional write param wins unless the user wants the RCNN-style migration.

## Case study: corrosion grouping cross-bucket (TML / branch work)

**Bug:** `match_clip_data_files` listed under `s3_location`, but `run_grouping_pipeline` / `run_low_confidence_pipeline` used `s3_bucket=output_location.bucket` for both reads and writes.

**Correct minimal fix:**

```python
run_kwargs = dict(
    s3_bucket=s3_location.bucket,              # reads
    s3_output_bucket=output_location.bucket,   # writes
    s3_output_key=s3_key,
)
```

In pipelines: `write_bucket = s3_output_bucket or s3_bucket`; pass `s3_bucket` to readers; use `write_bucket` for list/upload.

**Detours to avoid next time:**

1. `s3_input_bucket` while leaving `s3_bucket` as write (same split, worse name).
2. Stamping `bucket` on segments + `segment_parquet_uri` without being asked.
3. Expanding `_resolve_grouping_io` as the place dual-bucket “lives” instead of plain path DRY.
4. Extra ilipyutils tests that only re-prove the kwargs contract already covered at the corrosion call site.

**Short version to remember:** solve the failure mode the user named; do not solve an API-naming or architecture-purity problem first.

## How agents should use this file

- Read this before proposing multi-file S3 / I/O / “self-describing” refactors in ILI workflows.
- If the user’s first prompt already states the coupling, quote it back and implement that coupling break before any alternate design.
- If you catch yourself adding a second bucket param *or* changing discovery schemas for a write-path bug, stop and re-check this doc.
