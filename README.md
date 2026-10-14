# Breast Cancer Classification Pipeline

This project turns a notebook experiment into a small, reproducible Python machine
learning application. It classifies breast cancer observations as `malignant` or
`benign` and keeps preprocessing and prediction consistent by saving one fitted
scikit-learn Pipeline.

## Dataset and model

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset built into
scikit-learn. It contains 569 samples, 30 numeric features, and two classes:

- `0`: malignant
- `1`: benign

The model is Logistic Regression. A `StandardScaler` is placed before the model in
the same scikit-learn Pipeline. The scaler is fitted only on the training data,
which prevents data leakage and ensures that saved-model predictions use the same
preprocessing as training.

The notebook records the exploration and validation work, including a stratified
80/20 train/test split, classification metrics, a confusion matrix, 5-fold
cross-validation, sample prediction, save/reload validation, and invalid-input
checks. The recorded baseline test accuracy is 0.9825 and the mean cross-validation
accuracy is 0.9802.

## Project structure

```text
.
|-- notebooks/
|   `-- 01_exploration.ipynb   # Experimentation and validation record
|-- production_app/
|   |-- config.py              # Environment-based settings
|   |-- logging_config.py      # Application logging setup
|   |-- data.py                # Dataset loading and splitting
|   |-- preprocessing.py       # Reusable preprocessing steps
|   |-- training.py            # Pipeline construction and fitting
|   |-- evaluation.py          # Classification metrics and reports
|   |-- model_service.py       # Save, load, and predict helpers
|   |-- main.py                # Application orchestration
|   `-- __init__.py
|-- models/                    # Generated model artifacts
|-- run.py                     # Command-line entry point
|-- .env.example               # Safe configuration example
|-- .pre-commit-config.yaml    # Ruff and Black hooks
|-- pyproject.toml             # Project metadata and dependencies
|-- uv.lock                    # Reproducible dependency lockfile
`-- README.md
```

## Prerequisites

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)

## Installation and configuration

From the project root, sync the locked environment:

```powershell
uv sync
```

Create the local environment file from the safe example:

```powershell
Copy-Item .env.example .env
```

The supported settings are:

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | Application environment name |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MODEL_PATH` | `models/breast_cancer_pipeline.joblib` | Saved pipeline location |
| `TEST_SIZE` | `0.2` | Test-set fraction |
| `RANDOM_SEED` | `42` | Reproducible split and model seed |

Do not commit `.env`, trained model artifacts, or the virtual environment.

## Running the application

Train, evaluate, and save the fitted pipeline:

```powershell
uv run python run.py train
```

After training, load the saved pipeline and predict one sample. The input must
contain exactly 30 numeric features. This example uses the first sample from the
scikit-learn dataset:

```powershell
uv run python run.py predict --features '[17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]'
```

The prediction command loads the saved artifact and does not retrain the model.
For multiple samples, pass a JSON list of 30-feature lists instead.

## Notebook

Open `notebooks/01_exploration.ipynb` to review the original experiments. The
notebook is the experimentation record; the modules in `production_app/` are the
source of truth for the runnable application.

## Code quality

Run the checks locally:

```powershell
uv run ruff check .
uv run black .
uv run pre-commit install
uv run pre-commit run --all-files
```

The repository uses Ruff for linting and Black for formatting. Ruff also checks
the notebook; Black formats the Python source files unless its optional Jupyter
support is installed.
