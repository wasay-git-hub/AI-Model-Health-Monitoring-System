from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
from src.model_pipeline.evaluation import get_rmspe
from src.utils import load_params

config = load_params()

def tune_hyperparameters(X_train, y_train, model_type, model):
    
    if model_type == "Random Forest":
        hyperparams = config['tuning']['param_grid_RF']

    elif model_type == "XGBoost":
        hyperparams = config['tuning']['param_grid_XGB']

        # Linear Regression doesn't have hyperparameters to tune
    else:
        print("Model Type not supported for tuning!")
        return model

    tscv = TimeSeriesSplit(n_splits=config['tuning']['random_search']['cv_splits'])
    rmspe_scorer = make_scorer(get_rmspe, greater_is_better=False)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions= hyperparams,
        scoring= rmspe_scorer,
        cv= tscv,
        n_iter= config['tuning']['random_search']['n_iter'],
        n_jobs= config['tuning']['random_search']['n_jobs'],
        random_state= config['tuning']['random_search']['random_state'],
        verbose= 2
    )

    search.fit(X_train, y_train)
    print(f" Best Params: {search.best_params_}")
    print(f" Best Score: {-search.best_score_}") # It returns negative MAE, so we flip it

    # returns the best model
    return search.best_estimator_