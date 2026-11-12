# 02: CLI guards for the two untested entry points

**What to build:** The primary model run script and the sample-verification script fail the way the other entry points already do. Pointed at a feature table that does not exist, each prints the missing path plus the command that builds it, and exits non-zero without a traceback. Both guards are covered by tests so these are no longer the only untested command-line entry points in the project.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The primary run script exits 1 with the missing path and the build command when its feature table is absent.
- [x] The sample-verification script exits 1 with the missing path and the build command when the PR feature table is absent.
- [x] Neither guard path attempts to read or train.
- [x] Both guards are tested by loading the script and calling its entry point, following the pattern of the existing CLI tests.
- [x] The suite stays fast: no test in this ticket trains a model.

## Comments

Both guards already met every criterion, so the work was the missing tests.
`tests/test_run_primary_model_cli.py` and `tests/test_verify_pr_samples_cli.py`
follow the importlib-plus-`main` pattern: each points at a nonexistent table
under `tmp_path` and asserts exit 1, the missing path, the build command, no
traceback, and nothing written; neither trains. Landed as `05bb097` on
`spec/review-cleanup-and-py311-rerun` (branch `ticket/02-cli-guards`).
