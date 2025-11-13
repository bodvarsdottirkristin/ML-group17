# EXERCISE ORIGIN NOTICE (Global):
# - This script assembles classification code strictly from the DTU 02452 exercises you attached.
# - For each code section, we add a CITATION tag pointing to the exercise file and cell/section where the pattern originates,
#   and a CHANGES tag describing any adaptation (e.g., variable names, function wrapping, dataset path).
#
# Core sources:
# - 02452_exercise2_fall25/exercise2.ipynb: OneHotEncoder and label encoding usage hints
# - 02452_exercise4_fall25: Standardization formula (mean/std) applied within CV
# - 02452_exercise6_fall25/week6/exercise6.ipynb: Nested (two-level) cross-validation structure (outer/inner)
# - 02452_exercise7_fall25/exercise7.ipynb: StratifiedKFold + LogisticRegression usage and correlated t-test (Setup II)
# - 02452_exercise8_fall25/exercise8.ipynb (+ solution.html): Error rate computation and standardize-inside-fold guidance
# - 02452_exercise9_fall25/exercise9.ipynb: PyTorch ANN (Sequential, Sigmoid, BCELoss, SGD) and training loop
#
# Anything not present verbatim in the exercises is explicitly labeled under CHANGES.
import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any

# From exercise 7 (02452_exercise7_fall25/exercise7.ipynb): StratifiedKFold usage and metrics/confusion utilities.
# Changes: only imported the specific classes used here.
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder  # From exercise 2: OneHotEncoder hint and usage

# From exercise 9 (02452_exercise9_fall25/exercise9.ipynb): Torch model/train loop patterns with BCE loss and SGD.
# Changes: adapted to a utility function and our dataset, kept 1 hidden layer.
import torch


# Plotting imports drawn from exercises that use matplotlib/seaborn/metrics
# CITATION:
#   - (Exercise 7) imports seaborn as sns; (Exercise 8) many matplotlib plots and error curves;
#   - (Exercise 9) uses matplotlib for decision boundary figures;
#   - (Exercise 7) uses sklearn.metrics/plots in tasks.
# CHANGES:
#   - Consolidate imports here for plotting utilities.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Anchor to avoid unused-import warnings in some linters when plots are optional at runtime
_PLOT_IMPORTS_ANCHOR = (plt, sns, metrics)


# -----------------------------
# Utility: data loading
# -----------------------------
def load_saheart_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load SAHeart.csv, keep features for classification and y=chd.
    - Manual source alignment: Project2 manual specifies CHD as target.
    - We will treat 'famhist' as categorical for one-hot (exercise 2 approach).

    CITATION:
      - (Exercise 2) OneHotEncoder/LabelEncoder hints, structure for X/y split
    CHANGES:
      - Adapted to SAHeart columns; drops 'ldl' for classification.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # Keep classification features (drop 'ldl' which belongs to regression part)
    X = df.drop(columns=["chd", "ldl"])
    y = df["chd"].astype(int)
    return X, y


# -----------------------------
# Utilities: preprocessing (inside CV only)
# -----------------------------
def split_feature_types(X_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Split columns into numeric and categorical lists.
    """
    numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def fit_preprocessor(
    X_train_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:
    """
    Fit standardization stats for numeric and OneHotEncoder for categoricals.
    - OneHotEncoder usage from exercise 2 (exercise2.ipynb Task 3.7 hint/solution).
    - Standardization formula from exercise 4 (exercise4.ipynb): (x - mean) / std with ddof=1.
      Changes: apply only to numeric columns; add 1e-8 epsilon for stability.

    CITATION:
      - (Exercise 2) OneHotEncoder usage and get_feature_names_out pattern
      - (Exercise 4) Standardization (mean/std with ddof=1)
    CHANGES:
      - Wrapped into a reusable function so it can be called inside folds.
      - Added epsilon to avoid division by zero.
    """
    state: Dict[str, Any] = {}

    if numeric_cols:
        mu = X_train_df[numeric_cols].mean(axis=0).values
        std = X_train_df[numeric_cols].std(ddof=1, axis=0).values + 1e-8
        state["numeric_cols"] = numeric_cols
        state["mu"] = mu
        state["std"] = std
    else:
        state["numeric_cols"] = []
        state["mu"] = np.array([])
        state["std"] = np.array([])

    if categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(X_train_df[categorical_cols])
        state["categorical_cols"] = categorical_cols
        state["ohe"] = ohe
        state["ohe_feature_names"] = list(ohe.get_feature_names_out(categorical_cols))
    else:
        state["categorical_cols"] = []
        state["ohe"] = None
        state["ohe_feature_names"] = []

    return state


def transform_with_preprocessor(
    X_df: pd.DataFrame,
    state: Dict[str, Any],
) -> Tuple[np.ndarray, List[str]]:
    """
    Transform features using fitted state (no refit).
    - Standardization and OneHotEncoder application per exercise 2 & 4, applied inside CV folds (exercise 8: standardize inside folds).
      Changes: Concatenate [numeric_std | ohe_cats] and return numpy array.

    CITATION:
      - (Exercise 8) Standardize within folds; do not leak validation info.
    CHANGES:
      - Concatenation and feature_names return for compatibility.
    """
    parts: List[np.ndarray] = []
    feature_names: List[str] = []

    if state["numeric_cols"]:
        Xn = X_df[state["numeric_cols"]].values
        Xn_std = (Xn - state["mu"]) / state["std"]
        parts.append(Xn_std.astype(np.float32))
        feature_names.extend(state["numeric_cols"])

    if state["categorical_cols"]:
        Xc = X_df[state["categorical_cols"]]
        Xc_ohe = state["ohe"].transform(Xc)
        parts.append(Xc_ohe.astype(np.float32))
        feature_names.extend(state["ohe_feature_names"])

    if parts:
        X_out = np.concatenate(parts, axis=1)
    else:
        # No features edge-case
        X_out = np.zeros((len(X_df), 0), dtype=np.float32)
    return X_out, feature_names


# -----------------------------
# Baseline (train-majority)
# -----------------------------
def majority_class_predictor(y_train: np.ndarray, n: int) -> np.ndarray:
    """
    Baseline: predict majority class of training set.
    - Error rate calculation pattern from exercise 8 (exercise8.ipynb): (y != y_est).sum() / N.

    CITATION:
      - (Exercise 8) Error rate definition and computation
    CHANGES:
      - Implemented a majority class baseline per project description.
    """
    vals, counts = np.unique(y_train, return_counts=True)
    maj = vals[np.argmax(counts)]
    return np.full((n,), maj, dtype=int)


# -----------------------------
# Logistic Regression (L2)
# -----------------------------
def tune_logistic_lambda_inner_cv(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    lambdas: np.ndarray,
    K_inner: int,
    seed: int,
) -> Tuple[float, Dict[str, Any]]:
    """
    Tune L2 strength (lambda) via inner StratifiedKFold CV.
    - Inner/Outer CV structure adapted from exercise 6 (week6/exercise6.ipynb Part 4 nested CV).
    - Standardize inside inner folds (exercise 8 Task 2 guidance).
    - Error rate per exercise 8.
    Returns best_lambda and an example fitted outer preprocessor (None here).

    CITATION:
      - (Exercise 6) Nested CV pattern (outer/inner KFold loops)
      - (Exercise 7) LogisticRegression usage (+ StratifiedKFold import)
      - (Exercise 8) Error rate metric
    CHANGES:
      - Classification (error rate) instead of MSE; parameterized lambda grid.
      - Used C=1/lambda mapping for scikit-learn logistic.
    """
    skf_inner = StratifiedKFold(n_splits=K_inner, shuffle=True, random_state=seed)

    mean_errors: List[float] = []
    for lam in lambdas:
        fold_errs: List[float] = []
        for inner_train_idx, inner_val_idx in skf_inner.split(X_train_df, y_train):
            X_tr_df = X_train_df.iloc[inner_train_idx]
            X_va_df = X_train_df.iloc[inner_val_idx]
            y_tr = y_train[inner_train_idx]
            y_va = y_train[inner_val_idx]

            # Fit preprocessor on inner-train only (exercise 8)
            state = fit_preprocessor(X_tr_df, numeric_cols, categorical_cols)
            X_tr, _ = transform_with_preprocessor(X_tr_df, state)
            X_va, _ = transform_with_preprocessor(X_va_df, state)

            # From exercise 7 logistic usage (Section 6.3 hints): LogisticRegression().
            # Changes: set C = 1/lambda for L2; solver lbfgs; max_iter increased for convergence.
            C = 1.0 / lam if lam > 0 else 1e6
            clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
            clf.fit(X_tr, y_tr)
            y_hat = clf.predict(X_va)
            err = float(np.sum(y_hat != y_va)) / len(y_va)
            fold_errs.append(err)
        mean_errors.append(float(np.mean(fold_errs)))

    best_idx = int(np.argmin(mean_errors))
    best_lambda = float(lambdas[best_idx])
    return best_lambda, {}


def fit_eval_logistic_once(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    lam: float,
) -> float:
    """
    Fit Logistic with chosen lambda on outer-train and evaluate error on outer-test.
    """
    state = fit_preprocessor(X_train_df, numeric_cols, categorical_cols)
    X_tr, _ = transform_with_preprocessor(X_train_df, state)
    X_te, _ = transform_with_preprocessor(X_test_df, state)
    C = 1.0 / lam if lam > 0 else 1e6
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
    clf.fit(X_tr, y_train)
    y_hat = clf.predict(X_te)
    err = float(np.sum(y_hat != y_test)) / len(y_test)
    return err


# -----------------------------
# ANN (1 hidden layer)
# -----------------------------
def build_ann_model(input_dim: int, hidden_units: int) -> torch.nn.Sequential:
    """
    From exercise 9 (exercise9.ipynb get_model): torch.nn.Sequential with Linear-ReLU-Linear-Sigmoid for binary classification.
    Changes: parameterized hidden_units.

    CITATION:
      - (Exercise 9) get_model(input_dim, hidden_dim, output_dim) with Sequential, ReLU, Sigmoid
    CHANGES:
      - Wrapped with explicit hidden_units argument; output_dim=1 for binary.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=input_dim, out_features=hidden_units, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=hidden_units, out_features=1, bias=True),
        torch.nn.Sigmoid(),
    )


def train_ann_binary(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_epochs: int = 1000,
    lr: float = 0.01,
    hidden_units: int = 8,
    seed: int = 0,
) -> torch.nn.Sequential:
    """
    Train simple ANN, return trained model.
    - Based on exercise 9 training loop with SGD and BCELoss.
    """
    torch.manual_seed(seed)
    model = build_ann_model(X_tr.shape[1], hidden_units)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32)

    for _ in range(n_epochs):
        y_pred = model(Xtr)
        loss = criterion(y_pred, ytr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def tune_ann_hidden_inner_cv(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    hidden_grid: List[int],
    K_inner: int,
    seed: int,
    n_epochs: int = 1000,
    lr: float = 0.01,
) -> int:
    """
    Tune hidden units H via inner StratifiedKFold CV.
    - Structure like logistic tuner; ANN code from exercise 9.
    """
    skf_inner = StratifiedKFold(n_splits=K_inner, shuffle=True, random_state=seed)
    mean_errors: List[float] = []

    for H in hidden_grid:
        fold_errs: List[float] = []
        for inner_train_idx, inner_val_idx in skf_inner.split(X_train_df, y_train):
            X_tr_df = X_train_df.iloc[inner_train_idx]
            X_va_df = X_train_df.iloc[inner_val_idx]
            y_tr = y_train[inner_train_idx]
            y_va = y_train[inner_val_idx]

            state = fit_preprocessor(X_tr_df, numeric_cols, categorical_cols)
            X_tr, _ = transform_with_preprocessor(X_tr_df, state)
            X_va, _ = transform_with_preprocessor(X_va_df, state)

            model = train_ann_binary(X_tr, y_tr, n_epochs=n_epochs, lr=lr, hidden_units=H, seed=seed)
            with torch.no_grad():
                y_prob = model(torch.tensor(X_va, dtype=torch.float32)).numpy().reshape(-1)
            y_hat = (y_prob >= 0.5).astype(int)
            err = float(np.sum(y_hat != y_va)) / len(y_va)
            fold_errs.append(err)
        mean_errors.append(float(np.mean(fold_errs)))

    best_idx = int(np.argmin(mean_errors))
    return int(hidden_grid[best_idx])


def compute_logistic_inner_profile(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    lambdas: np.ndarray,
    K_inner: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CITATION:
      - (Exercise 6) Nested CV loops; (Exercise 8) error computation; (Exercise 7) Logistic usage.
    CHANGES:
      - Returns per-λ mean and SE over inner folds for plotting.
    """
    skf_inner = StratifiedKFold(n_splits=K_inner, shuffle=True, random_state=seed)
    per_lambda_errors: List[List[float]] = [[] for _ in range(len(lambdas))]

    for inner_train_idx, inner_val_idx in skf_inner.split(X_train_df, y_train):
        X_tr_df = X_train_df.iloc[inner_train_idx]
        X_va_df = X_train_df.iloc[inner_val_idx]
        y_tr = y_train[inner_train_idx]
        y_va = y_train[inner_val_idx]
        state = fit_preprocessor(X_tr_df, numeric_cols, categorical_cols)
        X_tr, _ = transform_with_preprocessor(X_tr_df, state)
        X_va, _ = transform_with_preprocessor(X_va_df, state)
        for i, lam in enumerate(lambdas):
            C = 1.0 / lam if lam > 0 else 1e6
            clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
            clf.fit(X_tr, y_tr)
            y_hat = clf.predict(X_va)
            err = float(np.sum(y_hat != y_va)) / len(y_va)
            per_lambda_errors[i].append(err)

    means = np.array([np.mean(e) for e in per_lambda_errors])
    ses = np.array([np.std(e, ddof=1) / np.sqrt(len(e)) for e in per_lambda_errors])
    return np.log10(lambdas), means, ses


def compute_ann_inner_profile(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    hidden_grid: List[int],
    K_inner: int,
    seed: int,
    n_epochs: int = 1000,
    lr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CITATION:
      - (Exercise 9) ANN model and training loop; (Exercise 6/8) inner CV fold structure and error formulation.
    CHANGES:
      - Returns mean and SE vs hidden units for plotting.
    """
    skf_inner = StratifiedKFold(n_splits=K_inner, shuffle=True, random_state=seed)
    per_H_errors: Dict[int, List[float]] = {h: [] for h in hidden_grid}

    for inner_train_idx, inner_val_idx in skf_inner.split(X_train_df, y_train):
        X_tr_df = X_train_df.iloc[inner_train_idx]
        X_va_df = X_train_df.iloc[inner_val_idx]
        y_tr = y_train[inner_train_idx]
        y_va = y_train[inner_val_idx]
        state = fit_preprocessor(X_tr_df, numeric_cols, categorical_cols)
        X_tr, _ = transform_with_preprocessor(X_tr_df, state)
        X_va, _ = transform_with_preprocessor(X_va_df, state)
        for H in hidden_grid:
            model = train_ann_binary(X_tr, y_tr, n_epochs=n_epochs, lr=lr, hidden_units=H, seed=seed)
            with torch.no_grad():
                y_prob = model(torch.tensor(X_va, dtype=torch.float32)).numpy().reshape(-1)
            y_hat = (y_prob >= 0.5).astype(int)
            err = float(np.sum(y_hat != y_va)) / len(y_va)
            per_H_errors[H].append(err)

    means = np.array([np.mean(per_H_errors[h]) for h in hidden_grid])
    ses = np.array([np.std(per_H_errors[h], ddof=1) / np.sqrt(len(per_H_errors[h])) for h in hidden_grid])
    return means, ses

def fit_eval_ann_once(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    hidden_units: int,
    n_epochs: int = 1000,
    lr: float = 0.01,
    seed: int = 0,
) -> float:
    state = fit_preprocessor(X_train_df, numeric_cols, categorical_cols)
    X_tr, _ = transform_with_preprocessor(X_train_df, state)
    X_te, _ = transform_with_preprocessor(X_test_df, state)

    model = train_ann_binary(X_tr, y_train, n_epochs=n_epochs, lr=lr, hidden_units=hidden_units, seed=seed)
    with torch.no_grad():
        y_prob = model(torch.tensor(X_te, dtype=torch.float32)).numpy().reshape(-1)
    y_hat = (y_prob >= 0.5).astype(int)
    err = float(np.sum(y_hat != y_test)) / len(y_test)
    return err


# -----------------------------
# Setup II: correlated t-test (exercise 7)
# -----------------------------
def correlated_ttest(r: np.ndarray, rho: float, alpha: float = 0.05) -> Tuple[float, Tuple[float, float], float]:
    """
    From exercise 7 (exercise7.ipynb Part 5, correlated t-test function).
    Changes: direct numpy/scipy implementation as a function here.

    CITATION:
      - (Exercise 7) correlated t-test (Setup II) function signature and math
    CHANGES:
      - Packaged as standalone function for reuse; identical formulas.
    """
    import scipy.stats as st

    J = len(r)
    r_hat = float(np.mean(r))
    s_hat = float(np.std(r, ddof=1))
    sigma_tilde = s_hat * math.sqrt((1.0 / J) + (rho / (1.0 - rho)))
    CI = st.t.interval(1 - alpha, df=J - 1, loc=r_hat, scale=sigma_tilde)
    p = 2.0 * st.t.cdf(-abs(r_hat) / sigma_tilde, df=J - 1)
    return r_hat, (float(CI[0]), float(CI[1])), float(p)


# -----------------------------
# Nested CV pipeline
# -----------------------------
def run_nested_cv_classification(
    X_df: pd.DataFrame,
    y: np.ndarray,
    K_outer: int = 10,
    K_inner: int = 10,
    seed: int = 1234,
) -> Dict[str, Any]:
    numeric_cols, categorical_cols = split_feature_types(X_df)

    # Grids (exercise manual suggestions)
    lambdas = np.logspace(-4, 2, 15)  # Logistic grid
    hidden_grid = [1, 2, 4, 8, 16, 32]  # ANN grid (exercise 9)

    skf_outer = StratifiedKFold(n_splits=K_outer, shuffle=True, random_state=seed)

    results_rows: List[Dict[str, Any]] = []
    errors_base: List[float] = []
    errors_log: List[float] = []
    errors_ann: List[float] = []
    chosen_lams: List[float] = []
    chosen_Hs: List[int] = []

    # For pooled ROC across outer tests (manual appendix note; Exercise 7/metrics usage):
    pooled_true: List[int] = []
    pooled_prob_log: List[float] = []
    pooled_prob_ann: List[float] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf_outer.split(X_df, y), start=1):
        X_tr_df = X_df.iloc[tr_idx]
        X_te_df = X_df.iloc[te_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        # Baseline
        y_hat_base = majority_class_predictor(y_tr, len(y_te))
        err_base = float(np.sum(y_hat_base != y_te)) / len(y_te)

        # Logistic: tune lambda in inner CV
        best_lam, _ = tune_logistic_lambda_inner_cv(
            X_tr_df, y_tr, numeric_cols, categorical_cols, lambdas, K_inner, seed
        )
        err_log = fit_eval_logistic_once(
            X_tr_df, y_tr, X_te_df, y_te, numeric_cols, categorical_cols, best_lam
        )

        # ANN: tune H in inner CV
        best_H = tune_ann_hidden_inner_cv(
            X_tr_df, y_tr, numeric_cols, categorical_cols, hidden_grid, K_inner, seed, n_epochs=500, lr=0.01
        )
        err_ann = fit_eval_ann_once(
            X_tr_df, y_tr, X_te_df, y_te, numeric_cols, categorical_cols, best_H, n_epochs=500, lr=0.01, seed=seed
        )

        # Also store probabilities for pooled ROC (Exercise 7 metrics usage)
        # Logistic probs
        state_log = fit_preprocessor(X_tr_df, numeric_cols, categorical_cols)
        X_tr_log, _ = transform_with_preprocessor(X_tr_df, state_log)
        X_te_log, _ = transform_with_preprocessor(X_te_df, state_log)
        C = 1.0 / best_lam if best_lam > 0 else 1e6
        clf_log = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
        clf_log.fit(X_tr_log, y_tr)
        prob_log = clf_log.predict_proba(X_te_log)[:, 1]

        # ANN probs
        state_ann = fit_preprocessor(X_tr_df, numeric_cols, categorical_cols)
        X_tr_ann, _ = transform_with_preprocessor(X_tr_df, state_ann)
        X_te_ann, _ = transform_with_preprocessor(X_te_df, state_ann)
        model_ann = train_ann_binary(X_tr_ann, y_tr, n_epochs=500, lr=0.01, hidden_units=best_H, seed=seed)
        with torch.no_grad():
            prob_ann = model_ann(torch.tensor(X_te_ann, dtype=torch.float32)).numpy().reshape(-1)

        pooled_true.extend(list(y_te))
        pooled_prob_log.extend(list(prob_log))
        pooled_prob_ann.extend(list(prob_ann))

        results_rows.append(
            {
                "fold": fold_idx,
                "lambda_star": best_lam,
                "E_test_LOG": err_log,
                "H_star": best_H,
                "E_test_ANN": err_ann,
                "E_test_BASE": err_base,
                "n_test": int(len(y_te)),
            }
        )
        chosen_lams.append(best_lam)
        chosen_Hs.append(best_H)
        errors_base.append(err_base)
        errors_log.append(err_log)
        errors_ann.append(err_ann)

        print(
            f"[Outer {fold_idx:02d}] λ*={best_lam:.2e}  E_LOG={err_log:.3f} | "
            f"H*={best_H:<2d} E_ANN={err_ann:.3f} | E_BASE={err_base:.3f} (n_test={len(y_te)})"
        )

    # Statistical comparison (Setup II correlated t-test; exercise 7)
    # Paired differences r_j = err_A - err_B on each outer fold (positive means A worse than B).
    A_vs_B = {}
    rho = 1.0 / K_outer
    pairs = {
        "ANN_vs_LOG": (errors_ann, errors_log),
        "ANN_vs_BASE": (errors_ann, errors_base),
        "LOG_vs_BASE": (errors_log, errors_base),
    }
    for name, (a, b) in pairs.items():
        r = np.array(a) - np.array(b)
        r_hat, CI, p = correlated_ttest(r, rho=rho, alpha=0.05)
        A_vs_B[name] = {"r_hat": r_hat, "CI95": CI, "p_value": p}

    results_df = pd.DataFrame(results_rows)
    summary = {
        "mean_outer_errors": {
            "BASE": float(np.mean(errors_base)),
            "LOG": float(np.mean(errors_log)),
            "ANN": float(np.mean(errors_ann)),
        },
        "selected_lambdas": chosen_lams,
        "selected_Hs": chosen_Hs,
        "setupII_tests": A_vs_B,
        "pooled": {
            "y_true": np.array(pooled_true, dtype=int),
            "prob_log": np.array(pooled_prob_log, dtype=float),
            "prob_ann": np.array(pooled_prob_ann, dtype=float),
        },
    }
    return {"rows": results_df, "summary": summary}


def refit_logistic_for_interpretability(
    X_df: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    chosen_lambdas: List[float],
):
    """
    Refit Logistic on all data with a "reasonable" λ (median of λ_i*), per manual §9.

    CITATION:
      - (Project manual §9) Refit logistic with median λ for interpretability only.
    CHANGES:
      - Implements the refit and prints standardized coefficient magnitudes.
    """
    lam = float(np.median(chosen_lambdas)) if len(chosen_lambdas) else 1.0
    state = fit_preprocessor(X_df, numeric_cols, categorical_cols)
    X_all, feature_names = transform_with_preprocessor(X_df, state)
    C = 1.0 / lam if lam > 0 else 1e6
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
    clf.fit(X_all, y)
    coefs = clf.coef_.reshape(-1)
    intercept = float(clf.intercept_[0])

    # Print top coefficients by absolute value
    abs_idx = np.argsort(-np.abs(coefs))
    top_k = min(10, len(abs_idx))
    print("\nFinal logistic refit for interpretability only (median λ):")
    print(f"- λ_median = {lam:.2e}, C={1.0/lam if lam>0 else 1e6:.3g}")
    print(f"- Intercept (standardized space): {intercept:.6f}")
    print("- Top |standardized coefficients|:")
    for i in range(top_k):
        j = abs_idx[i]
        print(f"  {feature_names[j]:30s}  coef={coefs[j]: .6f}")

    print("\nHow logistic predicts (binary):")
    print("  Compute z = β0 + Σ_j β_j x_j_std; then p = σ(z) = 1/(1+exp(-z)). Predict 1 if p≥0.5 else 0.")
    return {"lambda_median": lam, "intercept": intercept, "coefs": coefs, "feature_names": feature_names}



# -----------------------------
# Plotting utilities (from exercises)
# -----------------------------
def plot_eda(X_df: pd.DataFrame, y: np.ndarray):
    """
    EDA plots:
    - Class balance bar (Exercise 7/8 plotting patterns with matplotlib/seaborn)
    - Per-feature class-conditional histograms for numeric features (Exercise 8 plotting style)
    - Correlation heatmap (Exercise 7 uses seaborn heatmap)
    """
    # CITATION:
    #   - (Exercise 8) plotting style with matplotlib/seaborn for distributions and comparisons
    #   - (Exercise 9) masking pattern: mask = (y == label) when plotting class-conditional views
    # CHANGES:
    #   - Use numpy boolean masks aligned by position to avoid pandas index alignment issues.
    y_series = pd.Series(y, name="chd")  # for class balance bar only
    sns.set_style("darkgrid")

    # Class balance
    plt.figure(figsize=(4, 3))
    y_series.value_counts().sort_index().plot(kind="bar", color=["tab:blue", "tab:red"])
    plt.title("Class balance (CHD)")
    plt.xlabel("CHD")
    plt.ylabel("Count")
    plt.tight_layout()
    # CITATION: Exercises call plt.show(); CHANGES: use non-blocking show so multiple figures can stay open
    plt.show(block=False)

    # Per-feature histograms (numeric only) by class
    numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    for col in numeric_cols:
        plt.figure(figsize=(4, 3))
        for cls, color in zip([0, 1], ["tab:blue", "tab:red"]):
            mask = (y == cls)  # CITATION: Exercise 8/9 re-plotting by class uses boolean mask on y
            plt.hist(X_df.loc[mask, col].dropna().values, bins=20, alpha=0.6, color=color, label=f"CHD={cls}")
        plt.title(f"{col} by class")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    # Correlation heatmap (numeric)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(6, 5))
        corr = X_df[numeric_cols].corr()
        # CITATION: (Exercise 7) seaborn heatmap usage
        # CHANGES: add annot=True and fmt to show values inside each cell
        sns.heatmap(corr, cmap="coolwarm", center=0, square=False, annot=True, fmt=".2f", annot_kws={"size": 7})
        plt.title("Correlation heatmap (numeric features)")
        plt.tight_layout()
        plt.show(block=False)


def plot_inner_cv_profiles(
    X_df: pd.DataFrame,
    y: np.ndarray,
    seed: int = 1234,
    K_inner: int = 10,
):
    """
    Inner-CV profiles:
    - error vs log10(λ) for logistic
    - error vs H for ANN
    CITATION:
      - (Exercise 6/8/7/9) for inner CV mechanics and plotting approach.
    """
    numeric_cols, categorical_cols = split_feature_types(X_df)
    lambdas_raw = np.logspace(-4, 2, 15)
    hidden_grid = [1, 2, 4, 8, 16, 32]

    # Logistic profile
    x_log, mean_log, se_log = compute_logistic_inner_profile(
        X_df, y, numeric_cols, categorical_cols, lambdas_raw, K_inner, seed
    )
    plt.figure(figsize=(5, 3))
    plt.errorbar(x_log, mean_log, yerr=se_log, fmt="o-", color="tab:blue")
    plt.xlabel("log10(lambda)")
    plt.ylabel("Validation error")
    plt.title("Inner-CV profile (Logistic)")
    plt.tight_layout()
    plt.show(block=False)

    # ANN profile
    mean_ann, se_ann = compute_ann_inner_profile(
        X_df, y, numeric_cols, categorical_cols, hidden_grid, K_inner, seed, n_epochs=300, lr=0.01
    )
    plt.figure(figsize=(5, 3))
    plt.errorbar(hidden_grid, mean_ann, yerr=se_ann, fmt="o-", color="tab:red")
    plt.xlabel("Hidden units H")
    plt.ylabel("Validation error")
    plt.title("Inner-CV profile (ANN)")
    plt.tight_layout()
    plt.show(block=False)


def plot_outer_diagnostics(results_df: pd.DataFrame):
    """
    Outer-fold diagnostics:
    - per-fold test errors (lines/dots) for BASE/LOG/ANN
    - bar chart with mean ± SE
    CITATION:
      - (Exercise 8) line/point charts and error bars; (Exercise 7) multi-model comparisons.
    """
    folds = results_df["fold"].values
    e_base = results_df["E_test_BASE"].values
    e_log = results_df["E_test_LOG"].values
    e_ann = results_df["E_test_ANN"].values

    plt.figure(figsize=(6, 3))
    plt.plot(folds, e_base, "o-", label="BASE", color="tab:gray")
    plt.plot(folds, e_log, "o-", label="LOG", color="tab:blue")
    plt.plot(folds, e_ann, "o-", label="ANN", color="tab:red")
    plt.xlabel("Outer fold")
    plt.ylabel("Test error")
    plt.title("Outer-fold test errors")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    means = [np.mean(e_base), np.mean(e_log), np.mean(e_ann)]
    ses = [
        np.std(e_base, ddof=1) / np.sqrt(len(e_base)),
        np.std(e_log, ddof=1) / np.sqrt(len(e_log)),
        np.std(e_ann, ddof=1) / np.sqrt(len(e_ann)),
    ]

    plt.figure(figsize=(4, 3))
    x = np.arange(3)
    labels = ["BASE", "LOG", "ANN"]
    plt.bar(x, means, yerr=ses, color=["tab:gray", "tab:blue", "tab:red"], alpha=0.8, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel("Test error")
    plt.title("Mean ± SE (outer folds)")
    plt.tight_layout()
    plt.show(block=False)


def plot_roc_curves(pooled: Dict[str, np.ndarray]):
    """
    Pooled ROC across outer test folds (manual appendix).
    CITATION:
      - (Exercise 7) metrics usage; ROC/AUC standard usage with sklearn.metrics.
    """
    y_true = pooled["y_true"]
    prob_log = pooled["prob_log"]
    prob_ann = pooled["prob_ann"]

    fpr_log, tpr_log, _ = metrics.roc_curve(y_true, prob_log)
    fpr_ann, tpr_ann, _ = metrics.roc_curve(y_true, prob_ann)
    auc_log = metrics.auc(fpr_log, tpr_log)
    auc_ann = metrics.auc(fpr_ann, tpr_ann)

    plt.figure(figsize=(4, 3))
    plt.plot(fpr_log, tpr_log, label=f"LOG (AUC={auc_log:.3f})", color="tab:blue")
    plt.plot(fpr_ann, tpr_ann, label=f"ANN (AUC={auc_ann:.3f})", color="tab:red")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Pooled ROC (outer tests)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show(block=False)


def plot_logistic_coefficients_bar(refit: Dict[str, Any]):
    """
    Coefficient magnitude plot for final logistic.
    CITATION:
      - (Exercise 8/7) bar/plot usage; (manual §9) interpretability bar plot.
    """
    coefs = refit["coefs"]
    names = refit["feature_names"]
    order = np.argsort(-np.abs(coefs))
    top_k = min(20, len(order))
    idx = order[:top_k]
    plt.figure(figsize=(6, 5))
    plt.barh([names[i] for i in idx][::-1], np.abs(coefs[idx])[::-1], color="tab:blue")
    plt.xlabel("|coef| (standardized)")
    plt.title("Final logistic: top |coef|")
    plt.tight_layout()
    plt.show(block=False)

def print_answers_report(X_df: pd.DataFrame, results: Dict[str, Any]):
    """
    Prints answers to the five required questions from 02452project_2.pdf.
    """
    rows = results["rows"]
    summary = results["summary"]

    print("\nQ1) Problem and type:")
    print("- We predict CHD from the remaining SA-Heart attributes. This is a binary classification problem (CHD ∈ {0,1}).")

    print("\nQ2) Models and hyperparameters:")
    print("- Baseline: train-majority (predicts the majority class of the TRAIN split).")
    print("- Logistic regression (L2): λ ∈ logspace(1e-4, 1e2); implemented via C=1/λ.")
    print("- Method 2 (ANN, 1 hidden layer): hidden units H ∈ {1,2,4,8,16,32}.")
    print("  Grids chosen per manual suggestions and exercise 9 patterns (small-capacity ANN).")

    print("\nQ3) Two-level CV results (reuse outer splits for all models):")
    print(rows.to_string(index=False))
    print("Mean outer-test errors:", summary["mean_outer_errors"])
    print("Brief discussion: ANN adds nonlinearity; logistic is strong baseline for linear separability;")
    print("baseline provides a sanity floor. Per-fold variability indicates dependence on split; nested CV mitigates selection bias.")

    print("\nQ4) Statistical comparison (Setup II, book §11.4; exercise 7):")
    for k, v in summary["setupII_tests"].items():
        CI = v["CI95"]
        print(f"- {k}: r_hat={v['r_hat']:.4f}, CI=({CI[0]:.4f},{CI[1]:.4f}), p={v['p_value']:.4g}")
    print("Interpretation: r_hat>0 means first method has higher error. Use p and CI to conclude superiority or parity.")

    print("\nQ5) Final logistic refit and explanation:")
    numeric_cols, categorical_cols = split_feature_types(X_df)
    refit_logistic_for_interpretability(X_df, results["y_all"], numeric_cols, categorical_cols, summary["selected_lambdas"])
    print("Compare with regression part: standardized |coef| magnitudes indicate feature influence; interpret with caution.")


def main():
    # Configuration per manual
    seed = 1234
    K_outer, K_inner = 10, 10
    data_path = os.path.join("data", "SAHeart.csv")

    # Ensure plotting imports considered used by linters (plots may be toggled)
    _ = _PLOT_IMPORTS_ANCHOR

    # CITATION: Exercises use plt.show(); CHANGES: enable interactive mode so all figures can open and remain visible
    plt.ion()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    X_df, y = load_saheart_dataset(data_path)
    out = run_nested_cv_classification(X_df, y.values, K_outer=K_outer, K_inner=K_inner, seed=seed)
    # add y_all to results for final refit (kept separate to keep provenance clear)
    out["y_all"] = y.values

    print("\nPer-fold results:")
    print(out["rows"].to_string(index=False))

    print("\nMean outer-test errors:")
    for k, v in out["summary"]["mean_outer_errors"].items():
        print(f"- {k}: {v:.4f}")

    print("\nSetup II correlated t-tests (r_hat, 95% CI, p-value):")
    for k, stats in out["summary"]["setupII_tests"].items():
        CI = stats["CI95"]
        print(f"- {k}: r_hat={stats['r_hat']:.4f}, CI=({CI[0]:.4f}, {CI[1]:.4f}), p={stats['p_value']:.4g}")

    # EDA plots
    plot_eda(X_df, y.values)

    # Inner-CV profiles (computed on full data as a visualization of tendencies)
    plot_inner_cv_profiles(X_df, y.values, seed=seed, K_inner=10)

    # Outer-fold diagnostics
    plot_outer_diagnostics(out["rows"])

    # ROC curves (pooled outer predictions)
    plot_roc_curves(out["summary"]["pooled"])

    # Print formal answers Q1-Q5
    print_answers_report(X_df, out)

    # Coefficient magnitude plot for final logistic refit
    numeric_cols, categorical_cols = split_feature_types(X_df)
    refit = refit_logistic_for_interpretability(X_df, out["y_all"], numeric_cols, categorical_cols, out["summary"]["selected_lambdas"])
    plot_logistic_coefficients_bar(refit)

    # Final blocking show to keep all figures visible
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

