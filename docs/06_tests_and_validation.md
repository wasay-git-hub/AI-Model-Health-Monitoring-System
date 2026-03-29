# Tests And Validation

## Current Test Suite
File: tests/test_pipeline.py

Test coverage intent:
- Verify configuration can be loaded.
- Verify cleaning logic removes closed stores and imputes CompetitionDistance with max.
- Verify feature engineering creates Year/Month/Day/WeekOfYear.
- Verify mapped categorical features become numeric.
- Verify split behavior creates non-empty and ratio-conforming outputs.

## Current Status
- tests/test_pipeline.py imports:
  - clean_data
  - feature_engineering
  - split_data
- Current preprocessing.py exports:
  - split_raw_data
  - process_data
  - extract_X_y
- Result: tests are API-misaligned and expected to fail unless preprocessing API is adapted or tests are rewritten.

## Validation Checklist For Future Changes
1. Config validation
- load_params returns expected top-level keys.

2. Data contract validation
- Required columns in train/store exist before merge and transformation.

3. Leakage control validation
- Statistics used for imputing CV/Test are derived from train only.

4. Split integrity validation
- Date ordering is strictly chronological across train -> cv -> test.

5. Metric validation
- RMSPE implementation handles zeros safely and consistently.

6. Persistence validation
- save_model writes artifact.
- load_model retrieves same artifact.

## Suggested Test Refactor Direction
- Replace legacy function imports in tests with current pipeline:
  - split_raw_data
  - process_data
  - extract_X_y
- Add end-to-end smoke test for run_pipeline with small fixture data.
- Add regression test for model selection decision (baseline vs tuned selection branch).

## Manual Validation Commands
- Run tests:
  - pytest tests/
- Run pipeline:
  - python src/main.py
- Run ANOVA utility:
  - python -m src.compare_models

If any command fails, compare against the API alignment notes in 07_known_issues_and_change_guidelines.md.
