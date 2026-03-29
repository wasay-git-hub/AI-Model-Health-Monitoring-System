# Notebook Reference

## notebooks/eda_analysis.ipynb

Purpose:
- Exploratory data analysis to understand target distribution, seasonality, nulls, and key business drivers.

Cell-by-cell functionality:
1. Load libraries and read data/train.csv + data/store.csv, print train.info().
2. Plot sales distribution histogram for Sales > 0.
3. Convert Date to datetime and plot monthly mean sales trend.
4. Visualize null patterns in store metadata with heatmap.
5. Plot Promo (0/1) vs Sales boxplot.
6. Merge train+store and plot StoreType vs Sales violin plot.
7. Scatter plot CompetitionDistance vs Sales.

Outputs expected:
- Multiple static diagnostic plots for feature understanding and data quality checks.

## notebooks/error_analysis.ipynb

Purpose:
- Analyze model residuals and failure modes after model training.

Cell-by-cell functionality:
1. Markdown title: ERROR ANALYSIS.
2. Load dependencies, add root path, load params, load+process data, split data, load best model artifact, predict on CV set, and create analysis dataframe with:
   - Actual_Sales
   - Predicted_Sales
   - Error
   - Abs_Error
   - Error_Percentage
3. Markdown: Top failures section.
4. Rebuild Date if needed, show top 10 worst prediction rows.
5. Markdown: systematic bias section.
6. Plot residual distribution histogram with zero reference line.
7. Markdown: failure mode section.
8. Plot Abs_Error by Promo and by StoreType.

Important compatibility note:
- The notebook imports preprocessing.clean_data, preprocessing.feature_engineering, preprocessing.split_data, but current preprocessing.py exposes split_raw_data/process_data/extract_X_y.
- Notebook code is currently aligned to an older API and likely requires refactor to execute on current source.

## Notebook Governance Recommendations
- Keep notebooks analytical, not source-of-truth business logic.
- If preprocessing APIs change, update notebook imports immediately.
- Prefer calling src functions over reimplementing transformations in notebooks.
