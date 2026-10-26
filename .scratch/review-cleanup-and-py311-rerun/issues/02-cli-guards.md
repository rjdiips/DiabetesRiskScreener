# 02: CLI guards for the two untested entry points

**What to build:** The primary model run script and the sample-verification script fail the way the other entry points already do. Pointed at a feature table that does not exist, each prints the missing path plus the command that builds it, and exits non-zero without a traceback. Both guards are covered by tests so these are no longer the only untested command-line entry points in the project.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The primary run script exits 1 with the missing path and the build command when its feature table is absent.
- [ ] The sample-verification script exits 1 with the missing path and the build command when the PR feature table is absent.
- [ ] Neither guard path attempts to read or train.
- [ ] Both guards are tested by loading the script and calling its entry point, following the pattern of the existing CLI tests.
- [ ] The suite stays fast: no test in this ticket trains a model.
