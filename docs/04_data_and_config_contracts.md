# Data And Config Contracts

## Dataset Contracts

### data/train.csv
Observed header:
- Store
- DayOfWeek
- Date
- Sales
- Customers
- Open
- Promo
- StateHoliday
- SchoolHoliday

Role:
- Transactional, date-level sales dataset used as base supervised learning data.

### data/store.csv
Observed header:
- Store
- StoreType
- Assortment
- CompetitionDistance
- CompetitionOpenSinceMonth
- CompetitionOpenSinceYear
- Promo2
- Promo2SinceWeek
- Promo2SinceYear
- PromoInterval

Role:
- Store metadata joined to training records by Store.

### data/train_80.csv and data/input_1.csv..input_4.csv
- Produced by input_split.py.
- train_80.csv: shuffled 80 percent subset.
- input_1..4.csv: equal partition of remaining holdout 20 percent.

## Config Contract: src/params.yaml

### mappings
- Encodes categorical tokens:
  - a,b,c,d mapped to 1..4
  - 0 and None mapped to 0
- Used for categorical_features transformation.

### categorical_features
- Current list: StoreType, Assortment, StateHoliday.
- Each value mapped through mappings lookup; unknown values default to 0.

### models
- type controls active training model.
- params_RF controls RandomForestRegressor baseline.
- params_XGB controls XGBRegressor baseline.

### training_data
- target: Sales
- features list currently includes:
  - Store
  - DayOfWeek
  - Promo
  - StateHoliday
  - SchoolHoliday
  - StoreType
  - Assortment
  - CompetitionDistance
  - Year
  - Month
  - Day

### data split ratios
- train_ratio: 0.7
- cv_ratio: 0.15
- test_ratio: 0.15 (implicitly remainder in code)

### paths
- train_dataset: data/train.csv
- store_dataset: data/store.csv
- model_1: models/RandomForest.pkl
- model_2: models/LinearRegression.pkl
- model_3: models/XGBoost.pkl

### tuning
- random_search settings for RandomizedSearchCV.
- param_grid_RF: random forest search space.
- param_grid_XGB: xgboost search space.

### threshold
- PERFORMANCE_THRESHOLD_RMSPE controls whether tuning is skipped or executed.

## Feature Processing Contract
From preprocessing.process_data:
- CompetitionDistance NaN -> train max distance
- CompetitionOpenSinceMonth NaN -> 0
- CompetitionOpenSinceYear NaN -> 0
- Promo2SinceWeek NaN -> 0
- Promo2SinceYear NaN -> 0
- PromoInterval NaN -> None
- Open == 0 rows removed
- Date expanded to Year, Month, Day, WeekOfYear

## Modeling Contract
- X matrix columns must match config.training_data.features after process_data.
- y vector must be config.training_data.target.
- Model artifact naming depends on exact model_type string.

## Dependency Contract
requirements.txt currently lists:
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, scipy
- joblib, pyYAML, pytest
- plus pathlib, os, sys (stdlib modules, generally not required in pip requirements)
