# Known Issues And Change Guidelines

## Confirmed Known Issues

1. Preprocessing API mismatch across project
- Current preprocessing.py API:
  - split_raw_data
  - process_data
  - extract_X_y
- Legacy callers still used in:
  - src/compare_models.py
  - tests/test_pipeline.py
  - notebooks/error_analysis.ipynb
- Legacy function names referenced there:
  - clean_data
  - feature_engineering
  - split_data

2. Potential XGBoost parameter naming issue
- src/model.py passes sub_sample for XGBRegressor.
- Typical XGBoost sklearn API uses subsample.
- This can cause runtime errors depending on xgboost version.

3. requirements.txt includes stdlib names
- os, sys, pathlib should not usually be pip-installed dependencies.

4. Import duplication in main.py
- load_and_merge imported from both src.preprocessing and src.data_loader.
- The second import from src.data_loader overwrites the first symbol.

## Decision Framework For Future Changes

When modifying preprocessing behavior:
- Update src/preprocessing.py first.
- Immediately synchronize:
  - tests/test_pipeline.py
  - src/compare_models.py
  - notebooks/error_analysis.ipynb

When modifying feature list or mappings:
- Update src/params.yaml.
- Verify process_data still produces all configured features.
- Re-run pipeline and tests.

When modifying model choices:
- Keep model_type string exact across:
  - params.yaml
  - model.py
  - model_serializer.py
- Ensure model artifact filename conventions remain consistent.

When adjusting threshold/tuning policy:
- Check both branches:
  - tuning skipped branch
  - tuning executed branch
- Validate final best_model selection and test metrics.

## Safe Change Sequence (Recommended)
1. Read docs/02_end_to_end_pipeline.md and docs/03_module_reference.md.
2. Edit config and code in small increments.
3. Update tests/notebooks that import changed APIs.
4. Run pytest and pipeline.
5. Update docs files with any new behavior.

## Documentation Update Rule
Any code behavior change should be reflected in:
- docs/03_module_reference.md (API/function behavior)
- docs/04_data_and_config_contracts.md (contracts)
- docs/07_known_issues_and_change_guidelines.md (if risk profile changed)
