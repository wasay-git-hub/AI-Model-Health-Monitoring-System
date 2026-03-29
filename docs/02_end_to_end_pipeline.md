# End-to-End Pipeline Behavior

This file documents the exact logic in src/main.py.

## Entry
- Function: run_pipeline()
- Triggered when src/main.py is run as a script.
- Reads configuration via load_params().
- Uses:
  - train_dataset path from config.paths.train_dataset
  - store_dataset path from config.paths.store_dataset
  - threshold from config.PERFORMANCE_THRESHOLD_RMSPE

## Step-by-Step Execution
1. Load Data
- Calls load_and_merge(train_dataset, store_dataset).
- Returns merged dataframe keyed by Store.

2. Split Raw Data
- Calls split_raw_data(df).
- Splits chronologically using Date after sorting.
- Ratios from config.data.train_ratio and config.data.cv_ratio.
- Test set is remaining rows.

3. Process Data With Leakage Control
- Training set: process_data(train_raw, train_stats=None)
  - Computes max CompetitionDistance and stores as train_stats.max_dist.
- CV/Test sets: process_data(set, train_stats=train_stats)
  - Reuses train max_dist to avoid leakage.
- Calls extract_X_y for train/cv/test.

4. Train Baseline Model
- Reads model_type from config.models.type.
- Calls train_model(X_train, y_train, model_type).

5. Baseline CV Prediction
- Calls model_prediction(model, X_cv).

6. Baseline CV Evaluation
- Calls get_evaluations(y_cv, y_pred_default).
- Uses RMSPE as primary decision metric.

7. Conditional Hyperparameter Tuning
- If baseline RMSPE <= threshold: skip tuning.
- Else:
  - Calls tune_hyperparameters(X_train, y_train, model_type, model).
  - Predicts tuned model on CV.
  - Evaluates tuned model.
  - Keeps tuned model only if tuned RMSPE <= baseline RMSPE.

8. Final Test Evaluation
- Predicts with selected best_model on X_test.
- Runs get_evaluations(y_test, y_pred_final).

9. Persistence
- Calls save_model(best_model, model_type).
- Output path selected from params.yaml paths.model_1/model_2/model_3.

## Decision Logic Summary
- Tuning gate:
  - Enter tuning only when baseline fails threshold.
- Model selection after tuning:
  - Keep tuned model if RMSPE does not worsen.
  - Else keep baseline.

## Inputs and Outputs
- Inputs:
  - data/train.csv
  - data/store.csv
  - src/params.yaml
- Intermediate outputs:
  - In-memory split and transformed dataframes.
  - CV and test metric prints.
- Output artifact:
  - models/<ModelName>.pkl

## Operational Notes
- Pipeline assumes model artifact directory exists or is writable.
- Open==0 rows are dropped in all splits to avoid zero-division behavior for RMSPE.
- Categorical mapping applies StoreType, Assortment, StateHoliday.
