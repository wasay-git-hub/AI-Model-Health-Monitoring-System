# Rossmann Project Knowledge Base

Purpose: this documentation captures full current functionality of the project so future edits and design decisions can be made by reading docs first.

## Reading Order
1. 01_architecture_overview.md
2. 02_end_to_end_pipeline.md
3. 03_module_reference.md
4. 04_data_and_config_contracts.md
5. 05_notebooks_reference.md
6. 06_tests_and_validation.md
7. 07_known_issues_and_change_guidelines.md

## Full File Coverage Map
- Root
  - README.md: project narrative, model comparison summary, usage commands
  - requirements.txt: Python dependencies
  - input_split.py: utility script to create train_80.csv and 4 holdout input files
- data
  - train.csv: main training rows (sales transactions)
  - store.csv: store metadata
  - train_80.csv, input_1.csv..input_4.csv: derived split files
- src
  - __init__.py: package marker
  - main.py: primary orchestration entrypoint
  - data_loader.py: data ingestion + merge
  - preprocessing.py: split_raw_data, process_data, extract_X_y
  - model.py: model selection, training, prediction
  - optimization.py: randomized hyperparameter search with TimeSeriesSplit
  - evaluation.py: metrics calculation and logging
  - model_serializer.py: save/load model artifacts
  - compare_models.py: ANOVA model comparison script
  - utils.py: config + mappings loaders
  - params.yaml: all config, paths, mappings, tuning grids
- notebooks
  - eda_analysis.ipynb: exploratory visual analytics
  - error_analysis.ipynb: post-training error diagnostics
- tests
  - __init__.py: package marker
  - test_pipeline.py: preprocessing and split tests (currently API-misaligned with preprocessing.py)

## How To Use This Knowledge Base
- For changing behavior: start with 02_end_to_end_pipeline.md and 03_module_reference.md.
- For changing config/features/models: read 04_data_and_config_contracts.md first.
- For troubleshooting tests or breakages: read 06_tests_and_validation.md and 07_known_issues_and_change_guidelines.md.
