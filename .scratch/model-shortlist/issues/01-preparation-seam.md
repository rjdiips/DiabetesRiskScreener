# 01: Preparation seam

**What to build:** A public preparation step for the shared training pipeline. It applies the configured sample filters, resolves or loads the cluster-grouped split assignment, runs the existing cluster and household leakage checks, and returns the development frame, the test frame, the assignment, the row counts, and the split summary. The existing single-model pipeline uses this step and still writes the same artifacts as before. This is the point where development and test become separate objects, so later selection code can receive development rows without the test frame.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] A public preparation result exposes the development frame, test frame, split assignment, row counts, and split summary.
- [x] Development and test rows are disjoint, and the existing cluster and household overlap checks still run.
- [x] The existing single-model pipeline uses the preparation seam and writes the same artifact keys and metrics as before on a synthetic table.
- [x] A missing feature table, missing assignment, or failed sample filter produces a clear error.
- [x] Tests cover the preparation result on a synthetic cluster-grouped table and prove that every development respondent has a valid fold.
- [x] Existing pipeline tests continue to pass without changes to their public seam.

## Comments

Implemented on `feature/model-shortlist` (commits `bfcf0ac`, merge `dd7e6e7`; review fixes `790bab3`, merge `533dadb`).

`src/screener/pipeline.py` now exposes `PreparedData` and `prepare_training_data`. The pipeline consumes only `prepared.development` for selection and keeps test rows unreachable from that path. The seam validates every development fold, raises clear errors for an empty table, missing columns, a missing assignment file, incomplete assignment coverage, and filters that retain no rows. `run_training_pipeline` keeps its signature and artifact keys. `tests/test_pipeline.py` covers the partition, fold validity, and error paths. Full suite: 173 passed.
