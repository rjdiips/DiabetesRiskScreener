# 10: Seed-averaging capability for the selected winner

**What to build:** Provide the seed-averaging step as a reusable capability that ticket 11 applies to the winner of the selected measurement bundle. Refit the chosen candidate with ten seeds and average predicted probabilities. Fit the calibrator on pooled out-of-fold predictions and store it with the averaged model. Skip averaging when the winner is deterministic (logistic regression). Record the seed list, ensemble size, calibrator, threshold, feature schema, and model version in the artifact. Report mean and standard deviation across seeds. The capability must not assume the full measurement bundle; tests use a synthetic stochastic winner.

**Blocked by:** 05, 08

**Status:** resolved

- [x] The capability is family- and bundle-agnostic.
- [x] The winner is refit ten times only when its family is stochastic.
- [x] A deterministic winner produces a single model with no seed-averaging step.
- [x] Averaged probabilities are in [0, 1] and calibrate no worse than the best single seed on validation.
- [x] The artifact records seeds, ensemble size, calibrator, threshold, feature schema, and model version.
- [x] Tests cover the stochastic and deterministic paths without training on real data.
- [x] Ticket 11 invokes the capability on the selected bundle's winner.
