# API Deployment Tracker

## Rule
- Every action from now on will be logged here.
- Work will be executed one task at a time.

## Updated Main Goal
- Final demo must show all evaluation metrics for each input file and a comparison across input files.
- API design must support single file evaluation and multi file comparison.

## Metrics To Compare Across Files
1. RMSPE
2. MAE
3. RMSE
4. MAPE
5. R2

## Planned Endpoints
1. GET /health
- Service status and model readiness.

2. GET /model-info
- Model type, feature list, artifact path, version metadata.

3. POST /predict-health
- Input: one record payload.
- Output: predicted sales.

4. POST /evaluate-file
- Input: CSV file path or uploaded file with actual Sales column.
- Flow: preprocess input, run model, compute metrics including RMSPE.
- Output: RMSPE, MAE, RMSE, MAPE, R2 for that file.

5. POST /compare-input-files
- Input: list of input CSV file paths.
- Flow: evaluate each file and aggregate results.
- Output: full metrics table per file, rankings by each metric, and best/worst file per metric.

6. GET /metrics-history
- Output: latest evaluation and comparison runs for reporting.

## Step By Step Task List
1. Create tracking file and initialize plan. [DONE]
2. Add API serving dependencies to project requirements. [DONE]
3. Create API app entry file with FastAPI startup configuration. [DONE]
4. Create request and response schemas with strict validation. [DONE]
5. Implement model loader that loads once at startup. [DONE]
6. Add endpoint GET /health. [DONE]
7. Add endpoint GET /model-info. [DONE]
8. Add endpoint POST /predict-health. [DONE]
9. Add endpoint POST /evaluate-file to compute RMSPE for one input file. [DONE]
10. Add endpoint POST /compare-input-files to compare RMSPE across files. [DONE]
11. Add endpoint GET /metrics-history for recent comparisons. [DONE]
12. Add robust exception handling and consistent error responses. [DONE]
13. Add unit tests for endpoint success and failure cases. [DONE]
14. Add stress test script and benchmark instructions. [DONE]
15. Update README with API usage, examples, and metric comparison workflow. [DONE]

## Move Log
- 2026-03-28: Created this tracker file and initialized step by step plan.
- 2026-03-28: Updated plan for endpoint design to support per-file RMSPE and cross-file comparison reporting.
- 2026-03-28: Updated requirement to compare all model evaluation metrics (RMSPE, MAE, RMSE, MAPE, R2), not only RMSPE.
- 2026-03-28: Completed Step 2 by adding API dependencies (fastapi, uvicorn, python-multipart) to requirements.
- 2026-03-28: Completed Step 3 by creating src/api_app.py with FastAPI app setup and startup runtime state initialization.
- 2026-03-28: Completed Step 4 by creating src/api_schemas.py with strict Pydantic request and response schemas for prediction, file evaluation, file comparison, history records, and error payloads.
- 2026-03-28: Completed Step 5 by adding src/api_model_loader.py and wiring src/api_app.py to load the trained model once at startup and store model metadata in app state.
- 2026-03-28: Completed Step 6 by adding GET /health endpoint in src/api_app.py to report service status and model readiness.
- 2026-03-28: Completed Step 7 by adding GET /model-info endpoint in src/api_app.py to expose model type, artifact path, target, feature list, and metrics history path.
- 2026-03-28: Completed Step 8 by adding POST /predict-health endpoint in src/api_app.py to run single-record predictions using validated schema input.
- 2026-03-28: Completed Step 9 by adding src/api_metrics_service.py and POST /evaluate-file endpoint to preprocess one input CSV, run predictions, and compute RMSPE, MAE, RMSE, MAPE, and R2.
- 2026-03-28: Completed Step 10 by adding POST /compare-input-files endpoint with all-metric ranking support and shared multi-file comparison logic in src/api_metrics_service.py.
- 2026-03-28: Completed Step 11 by adding persistent JSONL metrics history storage (src/api_history.py), wiring evaluation/comparison endpoints to append records, and adding GET /metrics-history endpoint.
- 2026-03-28: Completed Step 12 by adding global FastAPI exception handlers in src/api_app.py for validation, HTTP errors, and unhandled exceptions with consistent JSON error payloads.
- 2026-03-28: Completed Step 13 by adding tests/test_api_endpoints.py covering success and validation-failure paths, and verified with pytest (7 passed).
- 2026-03-28: Completed Step 14 by adding scripts/stress_test_api.py for load testing and docs/api_stress_testing.md with benchmark profiles and bottleneck analysis workflow.
- 2026-03-28: Completed Step 15 by updating README.md with API startup instructions, endpoint usage examples, metrics comparison workflow, metrics history usage, and stress-testing command.
