# Module Reference

## src/utils.py

### load_params()
- Reads src/params.yaml from directory of utils.py.
- Returns parsed config dictionary.
- Raises FileNotFoundError if params.yaml missing.

### get_mappings()
- Reads src/params.yaml.
- Returns config['mappings'] dictionary.

## src/data_loader.py

### load_and_merge(train_path, store_path)
- Reads training and store CSVs.
- Left joins on Store.
- Returns merged dataframe.

## src/preprocessing.py

### load_and_merge(train_path, store_path)
- Duplicates function from data_loader.py.
- Same behavior: read and left-merge.

### split_raw_data(df)
- Sorts df by Date ascending.
- Splits by configured ratios:
  - train: [0, train_end)
  - cv: [train_end, val_end)
  - test: [val_end, end)
- Returns train_df, val_df, test_df.

### process_data(df, train_stats=None)
Cleaning:
- Casts StateHoliday to string.
- Computes max CompetitionDistance from train data (when train_stats is None).
- Fills CompetitionDistance with train max distance.
- Fills CompetitionOpenSinceMonth/Year with 0.
- Fills Promo2SinceWeek/Year with 0.
- Fills PromoInterval with "None".
- Drops rows where Open == 0.

Feature engineering:
- Converts Date to datetime.
- Adds Year, Month, Day, WeekOfYear.
- Maps configured categorical columns via mapping dictionary.

Returns:
- processed dataframe
- train_stats dictionary (currently max_dist)

### extract_X_y(df)
- Selects features from config.training_data.features.
- Selects target from config.training_data.target.
- Returns X, y.

## src/model.py

### train_model(X_train, y_train, model_type)
- model_type == "Random Forest": RandomForestRegressor with config params_RF.
- model_type == "Linear Regression": LinearRegression.
- model_type == "XGBoost": XGBRegressor with config params_XGB.
- Fits model on X_train, y_train.
- Returns fitted model.
- Raises ValueError for unsupported model type.

### model_prediction(model, X_val)
- Returns model.predict(X_val).

## src/evaluation.py

### get_rmspe(y_true, y_pred)
- Filters rows where y_true != 0.
- Computes RMSPE = sqrt(mean(((y_true - y_pred)/y_true)^2)).
- Rounds to 2 decimals.

### get_evaluations(y_true, y_pred)
- Computes and prints:
  - RMSPE (primary)
  - MAPE, MAE, RMSE (secondary)
  - R2 (statistical)
- Returns dictionary with MAE, RMSE, MAPE, RMSPE, R2.

## src/optimization.py

### tune_hyperparameters(X_train, y_train, model_type, model)
- Supports tuning for:
  - Random Forest: config.tuning.param_grid_RF
  - XGBoost: config.tuning.param_grid_XGB
- Linear Regression or unsupported types: returns input model unchanged.
- Uses TimeSeriesSplit with config-defined split count.
- Uses RandomizedSearchCV with custom RMSPE scorer.
- Returns search.best_estimator_.

## src/model_serializer.py

### save_model(model, model_type)
- Resolves model path by model_type from config paths.
- Dumps model with joblib.dump.

### load_model(model_type)
- Resolves artifact path by model_type.
- Raises FileNotFoundError if model missing.
- Loads model with joblib.load and returns it.

## src/compare_models.py

### run_model_comparison()
Intended behavior:
- Load and preprocess data.
- Split and obtain test data.
- Load saved model files.
- Predict each model on test data.
- Compare squared-error arrays via scipy.stats.f_oneway ANOVA.
- Print F statistic, p-value, and interpretation.

Important note:
- It imports clean_data, feature_engineering, split_data from preprocessing, but those symbols are not present in current preprocessing.py.
- This script currently does not match current preprocessing API and needs refactor to run.

## src/main.py

### run_pipeline()
- Full orchestrator documented in detail in 02_end_to_end_pipeline.md.

## tests/test_pipeline.py
- Contains five tests:
  - config load test
  - clean_data behavior test
  - feature_engineering output test
  - split_data shape test
- Current imports require clean_data, feature_engineering, split_data which do not exist in current preprocessing.py API.
- Tests are currently aligned to an older preprocessing design, not the active one.
