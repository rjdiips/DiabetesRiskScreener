# 05: One comparison accessor

**What to build:** The primary-versus-PR comparison module exposes one accessor for both a metric value within a named block (overall or survey-weighted) and a run-level field such as the calibrator or the selected model family. The two near-identical helpers collapse into it, call sites are updated, and the generated report is unchanged for unchanged inputs.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] One accessor returns metric values from a named block and defaults to the overall block.
- [ ] The same accessor returns run-level fields.
- [ ] A missing name returns the same value it returns today.
- [ ] The generated comparison report is byte-identical for unchanged inputs, verified against the committed report.
- [ ] Existing rendering tests pass without weakening their assertions.
- [ ] No new edge in the test seam: the report-level tests remain the proof of behaviour.
