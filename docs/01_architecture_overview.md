# Architecture Overview

## System Goal
Predict Rossmann daily sales using tabular historical data and store metadata with a configurable model pipeline.

## Core Design Principles In Current Code
- Modularization by responsibility (ingestion, preprocessing, modeling, evaluation, tuning, serialization).
- Centralized configuration in src/params.yaml.
- Time-aware split logic to reduce leakage (sort by Date, then chronological split).
- Conditional hyperparameter tuning based on an RMSPE threshold.

## Runtime Components
- Configuration Loader
  - src/utils.py loads params.yaml and categorical mappings.
- Data Layer
  - src/data_loader.py performs train/store CSV load + left join on Store.
  - src/preprocessing.py performs leakage-aware transforms and split logic.
- Model Layer
  - src/model.py instantiates one of 3 regressors.
  - src/optimization.py performs RandomizedSearchCV for RF/XGB.
- Evaluation Layer
  - src/evaluation.py computes RMSPE, MAPE, MAE, RMSE, R2.
- Persistence Layer
  - src/model_serializer.py stores/loads model binaries via joblib.
- Orchestration
  - src/main.py executes full training/evaluation/tuning/save flow.
- Statistical Comparison Utility
  - src/compare_models.py intended to run one-way ANOVA across saved models.

## Supported Model Types
- Linear Regression
- Random Forest
- XGBoost

Model selection is configuration-driven through models.type in src/params.yaml.

## Project Execution Modes
- Training pipeline mode: python src/main.py
- Statistical comparison mode: python -m src.compare_models
- Notebook analysis mode: manual execution in notebooks/

## Artifact Flow
1. Load source data from data/train.csv + data/store.csv.
2. Merge and preprocess.
3. Split into train/cv/test.
4. Train baseline model.
5. Evaluate on cv.
6. Optionally tune and re-evaluate.
7. Evaluate final model on test.
8. Save final model in models/*.pkl.

## Current Technical Risks
- API drift exists: compare_models.py and tests/test_pipeline.py use preprocessing function names not present in src/preprocessing.py.
- XGBoost parameter key mismatch in src/model.py (sub_sample) vs typical xgboost arg name (subsample).
- requirements.txt includes stdlib modules that should not be pip dependencies (os, sys, pathlib).
