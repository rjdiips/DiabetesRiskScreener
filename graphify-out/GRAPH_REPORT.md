# Graph Report - diabetes-risk-screener  (2026-11-12)

## Corpus Check
- 114 files · ~125,722 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 16 file(s) not represented in the graph (top: .csv 15, (none) 1)

## Summary
- 1010 nodes · 2708 edges · 44 communities (40 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 166 edges (avg confidence: 0.86)
- Token cost: 80,413 input · 59,613 output

## Community Hubs (Navigation)
- Split assignment and tests
- PR comparison reporting
- Variant comparison and run summary
- Spec concepts and scope
- Model shortlist spec and issues
- Calibration and threshold selection
- CLI scripts
- Training pipeline core
- Feature engineering
- Model bundle persistence
- PR sample reconciliation
- NFHS-5 data loading
- PR feature build and audit
- PR preparation seam tests
- Feature policy guards
- Model selection and refit
- Primary model config
- Config assembly and cleanup specs
- Metrics and calibration
- PR and primary model specs
- Feature tables and audit
- CLI script tests
- Metrics tests
- SHAP chart: PR 15-49 run
- Data definitions and tables
- Pipeline tests and entry point
- PR 15+ BP run and SHAP
- Feature cleaning tests
- PR results reports
- Config path helpers
- PR audit reconciliation
- PR calibration and label bias
- Model overview and splits
- Docs governance conventions
- SHAP chart: primary model
- Deployment and referral policy
- Model comparison and fairness
- Model shortlist vocabulary
- Comparison CLI tests
- Domain docs guidance
- JSON serialization helper
- NFHS-5 specification
- Project root

## God Nodes (most connected - your core abstractions)
1. `run_training_pipeline()` - 61 edges
2. `assign_pr_splits()` - 44 edges
3. `pr_15_plus_config()` - 40 edges
4. `TrainingConfig` - 37 edges
5. `build_pr_features()` - 31 edges
6. `_features()` - 31 edges
7. `pr_shaped_features()` - 28 edges
8. `TrainingResult` - 27 edges
9. `assign_splits()` - 26 edges
10. `compare_primary_vs_pr()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `Unweighted Primary Training with Survey-Weighted Evaluation` --references--> `TrainingConfig`  [INFERRED]
  docs/specs/india-diabetes-risk-screener-nfhs5-spec.md → src/screener/pipeline.py
- `Three Baselines and Validation-Only Model Selection` --rationale_for--> `run_training_pipeline()`  [INFERRED]
  .scratch/primary-model-pipeline/spec.md → src/screener/pipeline.py
- `12-Feature Default PR BP Set and Its Variants` --rationale_for--> `validate_feature_frame()`  [INFERRED]
  .scratch/pr-bp-sensitivity-model/spec.md → src/screener/feature_policy.py
- `Leakage Rules (Target-Derived and Design Variables)` --rationale_for--> `validate_feature_frame()`  [INFERRED]
  docs/specs/india-diabetes-risk-screener-nfhs5-spec.md → src/screener/feature_policy.py
- `Primary Model Training and Evaluation (spec)` --rationale_for--> `build_irmr_features()`  [INFERRED]
  .scratch/primary-model-pipeline/spec.md → src/screener/features.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three published runs on the shared training pipeline** — readme_run_training_pipeline, artifacts_primary_primary_irmr_15to49_20261026t083448602742z_42_4e9c2a9e_results_report, artifacts_pr_runs_pr_15plus_bp_20261026t084255023166z_42_2a9e78b6_results_report, artifacts_pr_runs_pr_15to49_bp_20261026t084051779902z_42_96adb799_results_report [EXTRACTED 1.00]
- **Subgroup and fairness reporting pattern** — comparison_subgroup_gaps, addendum_label_bias, primary_fairness_audit [INFERRED 0.75]
- **Primary-versus-PR comparison reporting flow** — artifacts_pr_comparison_primary_vs_pr_comparison, artifacts_pr_comparison_pr_sensitivity_model_card_addendum, artifacts_primary_primary_irmr_15to49_20261026t083448602742z_42_4e9c2a9e_results_report, artifacts_pr_runs_pr_15plus_bp_20261026t084255023166z_42_2a9e78b6_results_report, artifacts_pr_runs_pr_15to49_bp_20261026t084051779902z_42_96adb799_results_report [INFERRED 0.85]
- **Household-grouped training, ensembling, calibration, threshold selection** — docs_specs_india_diabetes_risk_screener_nfhs5_spec_household_grouped_splits, docs_specs_india_diabetes_risk_screener_nfhs5_spec_multi_seed_ensemble, docs_specs_india_diabetes_risk_screener_nfhs5_spec_probability_calibration, docs_specs_india_diabetes_risk_screener_nfhs5_spec_threshold_selection_80_sensitivity [INFERRED 0.85]
- **Launch serving surface: web pages, FastAPI service, predict endpoint, SHAP explanations** — docs_specs_india_diabetes_risk_screener_nfhs5_spec_web_app_pages, docs_specs_india_diabetes_risk_screener_nfhs5_spec_fastapi_service, docs_specs_india_diabetes_risk_screener_nfhs5_spec_api_predict_endpoint, docs_specs_india_diabetes_risk_screener_nfhs5_spec_shap_explainability [INFERRED 0.85]
- **Local markdown issue workflow: feature dir, spec, numbered issues, triage labels** — docs_agents_issue_tracker_scratch_feature_directory, docs_agents_issue_tracker_spec_md, docs_agents_issue_tracker_issue_file, docs_agents_triage_labels_triage_labels [INFERRED 0.75]
- **Preparation, selection evidence, and final evaluation phases** — _scratch_model_shortlist_spec_preparation_seam, _scratch_model_shortlist_spec_selection_evidence, _scratch_model_shortlist_spec_fit_handle, _scratch_model_shortlist_spec_final_evaluation [EXTRACTED 1.00]
- **Comparison runner to leakage-safe artifacts to seed-averaged freeze** — _scratch_model_shortlist_issues_08_comparison_runner_comparison_runner, _scratch_model_shortlist_issues_09_leakage_safe_comparison_artifacts_leakage_safe_comparison_artifacts, _scratch_model_shortlist_issues_10_seed_averaged_winner_seed_averaged_winner, _scratch_model_shortlist_issues_11_primary_comparison_and_freeze_primary_comparison_and_freeze [EXTRACTED 1.00]
- **PR Sensitivity Analysis Delivery Flow** — _scratch_pr_bp_sensitivity_model_spec_pr_feature_table, _scratch_pr_bp_sensitivity_model_spec_split_reuse, _scratch_pr_bp_sensitivity_model_issues_03_pr_15_plus_bp_model_run_pr_15_plus_bp_model_run, _scratch_pr_bp_sensitivity_model_spec_comparison_report, _scratch_pr_bp_sensitivity_model_spec_pr_sensitivity_model_card_addendum [INFERRED 0.85]
- **Primary Model Pipeline Decision Flow** — _scratch_primary_model_pipeline_spec_cluster_grouped_split_strategy, _scratch_primary_model_pipeline_spec_state_free_feature_policy, _scratch_primary_model_pipeline_spec_calibration, _scratch_primary_model_pipeline_spec_sensitivity_first_threshold, _scratch_primary_model_pipeline_spec_artifacts_and_model_bundle [INFERRED 0.85]
- **Review Cleanup Refactor Seams** — _scratch_review_cleanup_and_py311_rerun_spec_shared_config_assembly, _scratch_review_cleanup_and_py311_rerun_spec_shared_run_summary, _scratch_review_cleanup_and_py311_rerun_spec_single_reconciliation_rule, _scratch_review_cleanup_and_py311_rerun_spec_single_comparison_accessor [INFERRED 0.85]

## Communities (44 total, 4 thin omitted)

### Community 0 - "Split assignment and tests"
Cohesion: 0.05
Nodes (62): _resolve_assignment(), age_15_49_slice(), _allocate_counts(), assert_no_group_leakage(), _assert_partition_complete(), _assign_cluster_slots(), _assign_fold_slots(), assign_pr_splits() (+54 more)

### Community 1 - "PR comparison reporting"
Cohesion: 0.07
Nodes (68): Issue 12: Rerun PR sensitivity models and update published docs, Cleanup Issue 05: One Comparison Accessor, One Comparison Accessor for Metrics and Run Fields, _addendum_age_table(), _addendum_gap_table(), _addendum_metric_table(), _addendum_sample_table(), _addendum_threshold_table() (+60 more)

### Community 2 - "Variant comparison and run summary"
Cohesion: 0.08
Nodes (34): Cleanup Issue 01: Shared Run Summary and Row-Count Accessor, One Shared Run Summary Function, TrainingResult, _audit_rows(), _audit_table(), compare_pr_variants(), _deltas(), _feature_table() (+26 more)

### Community 3 - "Spec concepts and scope"
Cohesion: 0.05
Nodes (44): Child Ticket, Numbered Issue File (.scratch/.../issues/NN-slug.md), Local Markdown Issue Tracker, .scratch/<feature-slug>/ Feature Directory, .scratch/<feature-slug>/spec.md, Wayfinder Map (map.md), Wayfinding Operations, needs-info (+36 more)

### Community 4 - "Model shortlist spec and issues"
Cohesion: 0.10
Nodes (40): Issue 02: Candidate configuration, Issue 03: Metrics prefactor, Issue 04: Selection evidence seam, Issue 05: Fit handle and final evaluation, Issue 06: Add scored candidate families and train unweighted, Issue 07: Measurement, questionnaire-only, and ablation feature bundles, Issue 08: Comparison runner and ranking rule, Issue 09: Leakage-safe comparison artifacts (+32 more)

### Community 5 - "Calibration and threshold selection"
Cohesion: 0.10
Nodes (19): Out-of-Fold Isotonic vs Platt Calibration, _calibrator_scores(), _clean(), _confusion_at(), fit_calibrator(), IdentityCalibrator, IsotonicCalibrator, _logit() (+11 more)

### Community 6 - "CLI scripts"
Cohesion: 0.10
Nodes (9): main(), main(), _read_prefix(), main(), _read_assignment(), main(), main(), main() (+1 more)

### Community 7 - "Training pipeline core"
Cohesion: 0.10
Nodes (21): _audit_notes(), _feature_importance(), _fingerprint(), _format(), _gap(), _metrics_table(), _model_card(), _package_versions() (+13 more)

### Community 8 - "Feature engineering"
Cohesion: 0.10
Nodes (22): any_yes(), _base_ir(), _binary(), build_ir_features(), build_mr_features(), clean_cm(), clean_cooking_fuel_codes(), clean_diastolic() (+14 more)

### Community 9 - "Model bundle persistence"
Cohesion: 0.10
Nodes (9): Run Artifacts and Schema-Validated Model Bundle, ConstantModel, load_model_bundle(), ModelBundle, save_model_bundle(), _bundle(), test_bundle_predicts_calibrated_probabilities_and_the_threshold_flag(), test_bundle_rejects_a_frame_missing_a_training_feature() (+1 more)

### Community 10 - "PR sample reconciliation"
Cohesion: 0.11
Nodes (19): PR Sample Reconciliation and Definition of Done, pr_15_49_config(), pr_sample_summary(), reconcile_pr_15_49_sample(), reconcile_pr_bp_sample(), _sample(), test_pr_15_49_reconciliation_requires_the_flag_column(), test_pr_15_49_reconciliation_uses_the_age_15_49_flag() (+11 more)

### Community 11 - "NFHS-5 data loading"
Cohesion: 0.14
Nodes (15): _components(), main(), _summary(), build_target(), clean_glucose(), load_ir(), load_irmr_analysis(), load_mr() (+7 more)

### Community 12 - "PR feature build and audit"
Cohesion: 0.11
Nodes (16): build_pr_features(), cooking_fuel_clean(), features(), pr_frame(), _row(), test_cooking_fuel_mapping_covers_clean_dirty_and_missing_codes(), test_pr_audit_reconciles_against_matching_reference_counts(), test_pr_audit_reconciliation_reports_status_and_checks_without_a_summary() (+8 more)

### Community 13 - "PR preparation seam tests"
Cohesion: 0.18
Nodes (21): prepare_training_data(), pr_15_plus_config(), pr_shaped_features(), test_default_trial_order_is_unchanged(), test_fairness_variant_reports_the_household_head_audit_subgroups(), test_pipeline_loads_a_split_assignment_from_the_configured_path(), test_pipeline_rejects_a_target_column_as_a_feature(), test_pr_15_49_config_filters_the_same_partition_to_younger_adults() (+13 more)

### Community 14 - "Feature policy guards"
Cohesion: 0.11
Nodes (14): 32-Feature State-Free Primary Feature Policy, pr_feature_set(), primary_feature_set(), validate_feature_frame(), test_a_clean_feature_frame_passes(), test_each_bundle_is_the_full_set_minus_exactly_one_block(), test_every_primary_set_excludes_forbidden_and_target_derived_columns(), test_every_primary_set_passes_feature_frame_validation() (+6 more)

### Community 15 - "Model selection and refit"
Cohesion: 0.10
Nodes (9): build_model(), fit_model(), inference_matrix(), model_matrix(), predict_positive_probability(), _cross_validate_candidate(), _mean(), _select_and_refit() (+1 more)

### Community 16 - "Primary model config"
Cohesion: 0.13
Nodes (14): _sample_reference(), primary_15_49_config(), primary_sample_summary(), reconcile_primary_sample(), _sample(), test_primary_config_passes_no_categorical_columns(), test_primary_config_records_the_requested_set_as_the_feature_variant(), test_primary_config_records_the_run_role_and_default_feature_variant() (+6 more)

### Community 17 - "Config assembly and cleanup specs"
Cohesion: 0.14
Nodes (12): Subgroup and Fairness Audit, Cleanup Issue 02: CLI Guards for Untested Entry Points, Cleanup Issue 03: One Config Assembler for Primary and PR Models, Review Cleanup and Python 3.11 Rerun (spec), One Shared Config Assembly Path, assemble_config(), config_metadata(), shared_subgroups() (+4 more)

### Community 18 - "Metrics and calibration"
Cohesion: 0.17
Nodes (14): metrics_at_threshold(), binary_metrics(), _binned(), bootstrap_confidence_intervals(), _calibration_slope_intercept(), _clean(), expected_calibration_error(), _metric_function() (+6 more)

### Community 19 - "PR and primary model specs"
Cohesion: 0.17
Nodes (22): Issue 02: PR Split Reuse and Assignment, Issue 03: PR 15+ BP Model Run, Issue 04: PR 15-49 Age-Matched Run, Issue 05: PR Subgroups, Fairness, and Feature-Set Variants, Issue 06: Primary vs PR Comparison Report, Issue 07: Real-Data PR Runs and Publishable Aggregate Outputs, Issue 08: PR Sensitivity Model-Card Addendum, Primary vs PR Side-by-Side Comparison Report (+14 more)

### Community 20 - "Feature tables and audit"
Cohesion: 0.18
Nodes (10): _base_mr(), build_feature_audit(), build_irmr_features(), build_pr_feature_audit(), _missingness(), _read_stata(), _summary(), _validate_pr_input() (+2 more)

### Community 21 - "CLI script tests"
Cohesion: 0.14
Nodes (9): _load_script(), test_15_49_run_refuses_to_bypass_the_15_plus_split_assignment(), _load_script(), test_variant_suite_requires_the_saved_split_assignment(), test_variant_suite_writes_the_comparison_as_json_and_markdown(), _load_script(), test_run_reports_a_missing_feature_table_before_training(), _load_script() (+1 more)

### Community 22 - "Metrics tests"
Cohesion: 0.21
Nodes (13): net_benefit_table(), test_confusion_metrics_are_computed_at_the_given_threshold(), test_expected_calibration_error_matches_a_worked_two_bin_example(), test_net_benefit_table_covers_every_threshold_from_a_worked_example(), test_net_benefit_table_handles_no_negatives_without_dividing_by_zero(), test_net_benefit_table_handles_no_positives_without_dividing_by_zero(), test_net_benefit_table_keeps_its_shape_for_an_empty_sample(), test_net_benefit_table_uses_survey_weights_when_given() (+5 more)

### Community 23 - "SHAP chart: PR 15-49 run"
Cohesion: 0.13
Nodes (16): SHAP Global Feature Importance Bar Chart (run pr_15to49_bp_20261026T084051779902Z_42_96adb799), age (top-ranked feature, ~0.52 mean |SHAP|), Expected beeswarm (dot) format — absent from this rendering, direction of effect NOT shown, bp_medication (5th feature, ~0.042 mean |SHAP|), bp_readings (3rd feature, ~0.105 mean |SHAP|), diastolic_mean (4th feature, ~0.05 mean |SHAP|), education (8th feature, ~0.028 mean |SHAP|), female (6th feature, ~0.038 mean |SHAP|) (+8 more)

### Community 24 - "Data definitions and tables"
Cohesion: 0.20
Nodes (13): Fact-sheet target definition, Harmonised IR+MR feature table, NFHS-5 IR women's recode, NFHS-5 MR men's recode, NFHS-5 PR household-member recode, PR feature table, PR split assignment (data notes), PR-to-MR anthropometry merge validation (+5 more)

### Community 25 - "Pipeline tests and entry point"
Cohesion: 0.23
Nodes (11): Issue 01: Preparation seam, run_training_pipeline(), _shap_summary(), _write_frame(), _write_json(), test_candidate_labels_must_be_unique_and_non_empty(), test_configured_candidates_drive_the_trial_loop(), test_pinned_candidate_scores_and_fits_exactly_one_candidate() (+3 more)

### Community 26 - "PR 15+ BP run and SHAP"
Cohesion: 0.27
Nodes (14): PR 15+ BP Model Run pr_15plus_bp_20261026T084255023166Z_42_2a9e78b6, SHAP Summary (Mean |SHAP|) Chart, age (top-ranked feature, mean |SHAP| ~0.71), alcohol_any (mean |SHAP| ~0.002), bp_high (mean |SHAP| ~0.003), bp_medication (mean |SHAP| ~0.12), bp_readings (mean |SHAP| ~0.07), diastolic_mean (mean |SHAP| ~0.035) (+6 more)

### Community 27 - "Feature cleaning tests"
Cohesion: 0.15
Nodes (9): bmi_from_height_weight(), build_pr_anthropometry(), prepare_pr_men_anthropometry(), test_bmi_from_height_weight_uses_plausible_range(), test_clean_scaled_drops_missing_codes_and_out_of_range(), test_fact_sheet_target_uses_high_glucose_or_medication(), test_prepare_pr_men_anthropometry_cleans_and_falls_back_to_computed_bmi(), test_validate_pr_mr_merge_reports_full_match() (+1 more)

### Community 28 - "PR results reports"
Cohesion: 0.18
Nodes (10): Model card: pr_15plus_bp, pr_15plus_bp results report, Model card: pr_15to49_bp, pr_15to49_bp results report, Survey-weighted comparison metrics, 95% cluster-bootstrap intervals, pr_15plus_bp feature importance, pr_15to49_bp feature importance (+2 more)

### Community 29 - "Config path helpers"
Cohesion: 0.22
Nodes (5): main(), _run(), relative_to_project(), test_relative_to_project_leaves_outside_paths_absolute(), test_relative_to_project_renders_project_paths_relative()

### Community 30 - "PR audit reconciliation"
Cohesion: 0.24
Nodes (7): Issue 01: PR Feature Table and Audit, Harmonised PR Feature Table, Cleanup Issue 04: One Reconciliation Rule, One Declaration per Count, One Reconciliation Rule and One Declaration per Count, _compare_counts(), _observed_pr_counts(), reconcile_pr_audit()

### Community 31 - "PR calibration and label bias"
Cohesion: 0.22
Nodes (8): PR age gradient, PR calibration results, IIPS & ICF NFHS-5 citation (addendum), PR sensitivity model-card addendum, Reliability curves, IIPS & ICF NFHS-5 citation (data notes), IIPS & ICF NFHS-5 citation (README), PR sensitivity model-card addendum reference

### Community 32 - "Model overview and splits"
Cohesion: 0.29
Nodes (9): Cluster-grouped split, Household overlap check, Cluster-grouped validation split, FastAPI screener service, India Diabetes Risk Screener, PR BP-based sensitivity model, PR split assignment, Primary IR/MR 15-49 reference model (+1 more)

### Community 33 - "Docs governance conventions"
Cohesion: 0.32
Nodes (7): Agent instructions (AGENTS.md), Single-context domain docs convention, Issue tracker in .scratch/, Canonical triage labels, Project glossary vocabulary, pr_15plus_bp model family selection, primary_irmr_15to49 model family selection

### Community 34 - "SHAP chart: primary model"
Cohesion: 0.43
Nodes (8): SHAP Summary — Mean |SHAP| Feature Importance (Primary IRMR 15-49 Run), age — top-ranked feature (~0.46 mean |SHAP|), bmi — 2nd-ranked feature (~0.135 mean |SHAP|), Long tail: diet_*, bp_medication, marital_status, education, wealth, female, children_ever_born (<0.05 mean |SHAP|), mean(|SHAP value|) — average impact on model output magnitude, Model Run: primary / IRMR / ages 15-49 (2026-10-26T08:34:48Z), systolic_mean — 4th-ranked feature (~0.095 mean |SHAP|), waist_cm — 3rd-ranked feature (~0.115 mean |SHAP|)

### Community 35 - "Deployment and referral policy"
Cohesion: 0.38
Nodes (6): Deployed model, Referral decision, Screener app, Screening threshold, Seed-averaged ensemble, Winner

### Community 36 - "Model comparison and fairness"
Cohesion: 0.43
Nodes (5): Primary vs PR comparison report, Model card: primary_irmr_15to49, primary_irmr_15to49 results report, Subgroup gap tables, Primary religion and social-group fairness audit

### Community 37 - "Model shortlist vocabulary"
Cohesion: 0.33
Nodes (7): Candidate, Exploration bucket, Model family, Model variant, Promotion rule, Ranking rule, Shortlist

### Community 38 - "Comparison CLI tests"
Cohesion: 0.43
Nodes (4): _load_script(), test_cli_reports_a_missing_run_directory(), test_cli_writes_the_primary_vs_pr_comparison(), _write_run()

### Community 39 - "Domain docs guidance"
Cohesion: 0.40
Nodes (6): Flag ADR Conflicts Convention, docs/adr/ Architecture Decision Records, Domain Docs (Agent Guidance), /domain-modeling Skill, GLOSSARY-MAP.md (Multi-Context Index), GLOSSARY.md

## Ambiguous Edges - Review These
- `Issue tracker in .scratch/` → `Project glossary vocabulary`  [AMBIGUOUS]
  AGENTS.md · relation: conceptually_related_to
- `SHAP Global Feature Importance Bar Chart (run pr_15to49_bp_20261026T084051779902Z_42_96adb799)` → `Expected beeswarm (dot) format — absent from this rendering, direction of effect NOT shown`  [AMBIGUOUS]
  artifacts/pr/runs/pr_15to49_bp_20261026T084051779902Z_42_96adb799/shap_summary.png · relation: references
- `Model Run: primary / IRMR / ages 15-49 (2026-10-26T08:34:48Z)` → `age — top-ranked feature (~0.46 mean |SHAP|)`  [AMBIGUOUS]
  artifacts/primary/primary_irmr_15to49_20261026T083448602742Z_42_4e9c2a9e/shap_summary.png · relation: conceptually_related_to
- `mean(|SHAP value|) — average impact on model output magnitude` → `Long tail: diet_*, bp_medication, marital_status, education, wealth, female, children_ever_born (<0.05 mean |SHAP|)`  [AMBIGUOUS]
  artifacts/primary/primary_irmr_15to49_20261026T083448602742Z_42_4e9c2a9e/shap_summary.png · relation: conceptually_related_to

## Knowledge Gaps
- **46 isolated node(s):** `india-diabetes-risk-screener`, `Canonical triage labels`, `Household overlap check`, `Exploration bucket`, `95% cluster-bootstrap intervals` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 273 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_training_pipeline()` connect `Pipeline tests and entry point` to `Variant comparison and run summary`, `Spec concepts and scope`, `Model shortlist spec and issues`, `Calibration and threshold selection`, `CLI scripts`, `Training pipeline core`, `Model bundle persistence`, `NFHS-5 data loading`, `PR preparation seam tests`, `Model selection and refit`, `Metrics and calibration`, `PR and primary model specs`, `Config path helpers`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `run_training_pipeline()` (e.g. with `Issue 05: Fit handle and final evaluation` and `Three Baselines and Validation-Only Model Selection`) actually correct?**
  _`run_training_pipeline()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `india-diabetes-risk-screener`, `Canonical triage labels`, `Household overlap check` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Split assignment and tests` be split into smaller, more focused modules?**
  _Cohesion score 0.05483405483405483 - nodes in this community are weakly interconnected._
- **Why does `IR/MR harmonised feature table` connect `Data definitions and tables` to `Model overview and splits`, `CLI scripts`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Issue tracker in .scratch/` and `Project glossary vocabulary`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `SHAP Global Feature Importance Bar Chart (run pr_15to49_bp_20261026T084051779902Z_42_96adb799)` and `Expected beeswarm (dot) format — absent from this rendering, direction of effect NOT shown`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._