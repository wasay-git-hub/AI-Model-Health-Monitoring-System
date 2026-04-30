# SalesWatch: AI Model Health Monitoring System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Supabase](https://img.shields.io/badge/Cloud-Supabase-3ecf8e)
![Docker](https://img.shields.io/badge/Container-Docker-2496ed)
![CI/CD](https://img.shields.io/badge/Pipeline-GitHub--Actions-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

🔗 **Live Demo:** [saleswatch.onrender.com](https://sales-watch.onrender.com)

## Project Overview

SalesWatch is a full-stack **MLOps platform** designed for real-time sales forecasting and model health observability. Transitioning beyond a static notebook, this system operationalizes an **XGBoost** regression model into a containerized service to predict daily sales, accounting for multiple factors such as State Holidays, Competitor Store Distance, and Promotions.

The system acts as an **Intelligent Evaluator**, tracking predictive performance (RMSPE), operational health (Latency), and data integrity (Drift) across sequential chronological batches. By integrating cloud persistence and automated feedback loops, it simulates a production environment where model health is continuously audited.

The primary goal is not just to train a model, but to demonstrate **Operational Readiness** by transitioning a raw notebook into a production-grade codebase — specifically addressing the **Engineering Trade-offs** between model complexity, interpretability, and operational cost.

---

## Three-Tier Architecture

The system is architected to ensure complete separation of concerns:

1. **Persistence Layer:** Utilizing **Supabase (Cloud PostgreSQL)** via SQLAlchemy for long-term health logs and **MLflow** for experimental metadata.
2. **Logic Layer:** A high-performance **FastAPI** backend that manages model inference, batch evaluation, and asynchronous feedback processing.
3. **Presentation Layer:** A custom-built, responsive dashboard using **Vanilla HTML5, CSS3, and JavaScript**, providing real-time visualization of model health trends.

---

## Engineering Trade-offs & Model Selection

We evaluated three distinct model architectures to find the optimal balance between **Simplicity** and **Accuracy**.

| Model Architecture | Role | RMSPE (Test) | Trade-off Analysis |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Baseline | 0.46 | High interpretability and zero tuning, but failed to capture the non-linear seasonality of retail sales, resulting in a high RMSPE. |
| **Random Forest** | Challenger | 0.16 | Strong performance and handles outliers well, but produced large serialization artifacts (>100MB). Tuning reduced CV error by ~34% (0.56 → 0.37). |
| **XGBoost** | **Champion** | **0.15** | Achieved the highest accuracy. While more complex to tune, it offered the best latency-to-accuracy ratio for production. |

### Impact of Effective Hyperparameter Tuning

We implemented `RandomizedSearchCV` with **Time Series Split** to optimize models without data leakage. The results demonstrate that effective tuning is critical:

- **Random Forest:** Tuning reduced Validation RMSPE from **0.56 (Baseline)** to **0.37 (Tuned)**.
- **XGBoost:** Further refinement of gradient boosting parameters yielded a state-of-the-art result of **0.15** on the Test set.

### Statistical Validation (ANOVA)

To mathematically confirm our model selection, we performed a **One-Way ANOVA** test on the squared prediction errors of all three candidates on the Test set.

- **F-Statistic:** `9518.81`
- **P-Value:** `0.0000e+00` (p < 0.05)

**Conclusion:** The result is **statistically significant**, confirming we can reject the null hypothesis. The non-linear architecture of XGBoost provides a fundamental improvement over the baseline — not a random fluctuation.

---

## Key Features

- **Zero-Leakage Data Engine:** Implements a strict chronological split (Train 70 / CV 15 / Test 15) to prevent future data leakage.
- **Cloud Observability:** Automated logging of every inference and evaluation run to a cloud-native PostgreSQL instance.
- **Asynchronous Feedback Loop:** A dedicated `/feedback` endpoint allowing managers to back-fill ground truth data, triggering real-time re-evaluation of model "Health."
- **Modular Architecture:** Code is separated into logical modules (`preprocessing`, `model`, `evaluation`, `optimization`) within the `src/` package.
- **Reproducibility:** All hyperparameters, file paths, and split ratios are centralized in `src/params.yaml`.
- **Conditional Tuning:** The pipeline only triggers expensive hyperparameter tuning if the baseline model fails to meet the defined performance threshold.
- **Containerization:** Fully orchestrated via **Docker Compose** for consistent cross-environment deployment.
- **Automated CI/CD:** Integrated GitHub Actions pipeline that executes a **pytest** suite on every push, acting as a logic gate for production updates.

---

## CI/CD Pipeline

Every push to the repository triggers an automated GitHub Actions workflow that acts as a logic gate before any changes reach production.

The pipeline runs in two stages:

1. **Automated Testing:** Executes the full `pytest` suite covering schema integrity, preprocessing logic, and chronological leakage checks — alongside live API endpoint tests to verify inference and evaluation routes respond correctly.
2. **Automatic Deployment:** If and only if all tests pass, the latest changes are automatically deployed to **Render**, ensuring the production environment is always in a verified state.

No manual deployment steps are required.

---

## Project Structure

```text
AI-Model-Health-Monitoring-System/
├── .github/workflows/      # CI/CD Pipeline (GitHub Actions)
├── data/                   # Processed data batches & store metadata
├── models/                 # Serialized XGBoost champion model (.pkl)
├── notebooks/              # EDA & Deep Error Analysis
├── scripts/                # Stress testing & data splitting utilities
├── src/                    # Source Code
│   ├── fast_api/           # Logic Layer (API & DB Models)
│   ├── frontend/           # Presentation Layer (HTML/JS/CSS)
│   ├── model_pipeline/     # Training & Evaluation Engine
│   ├── params.yaml         # Centralized Configuration
│   └── utils.py            # Infrastructure loaders
├── tests/                  # Automated Test Suite (pytest and API tests)
├── Dockerfile              # Container blueprint
├── docker-compose.yml      # Service orchestrator
└── requirements.txt        # Managed dependencies
```

---

## Installation & Execution

### 1. Prerequisites

Due to GitHub file size limits, the trained model and data batches must be placed manually:

- Ensure `models/XGBoost.pkl` and `data/processed_input_*.csv` exist.
- Create a `.env` file with your `DATABASE_URL`.

### 2. Running via Docker (Recommended)

The entire system (API, Dashboard, and Monitoring) can be launched with a single command:

```bash
docker-compose up --build
```

| Service | URL |
| :--- | :--- |
| Dashboard | http://localhost:8000 |
| API Docs/SwaggerUI | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

### 3. Automated Testing

Run the logic gates manually to verify system integrity:

```bash
pytest tests/
```

---

## License

Academic Semester Project for **DS201 — Programming for AI**.