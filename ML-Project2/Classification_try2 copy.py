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


# -----------------------------
# Utility: data loading
# -----------------------------
def load_saheart_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load SAHeart.csv, keep features for classification and y=chd.
    - Manual source alignment: Project2 manual specifies CHD as target.
    - We will treat 'famhist' as categorical for one-hot (exercise 2 approach).
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
    X_va: np.ndarray,
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

            model = train_ann_binary(X_tr, y_tr, X_va, n_epochs=n_epochs, lr=lr, hidden_units=H, seed=seed)
            with torch.no_grad():
                y_prob = model(torch.tensor(X_va, dtype=torch.float32)).numpy().reshape(-1)
            y_hat = (y_prob >= 0.5).astype(int)
            err = float(np.sum(y_hat != y_va)) / len(y_va)
            fold_errs.append(err)
        mean_errors.append(float(np.mean(fold_errs)))

    best_idx = int(np.argmin(mean_errors))
    return int(hidden_grid[best_idx])


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

    model = train_ann_binary(X_tr, y_train, X_te, n_epochs=n_epochs, lr=lr, hidden_units=hidden_units, seed=seed)
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
    }
    return {"rows": results_df, "summary": summary}


def main():
    # Configuration per manual
    seed = 1234
    K_outer, K_inner = 10, 10
    data_path = os.path.join("data", "SAHeart.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    X_df, y = load_saheart_dataset(data_path)
    out = run_nested_cv_classification(X_df, y.values, K_outer=K_outer, K_inner=K_inner, seed=seed)

    print("\nPer-fold results:")
    print(out["rows"].to_string(index=False))

    print("\nMean outer-test errors:")
    for k, v in out["summary"]["mean_outer_errors"].items():
        print(f"- {k}: {v:.4f}")

    print("\nSetup II correlated t-tests (r_hat, 95% CI, p-value):")
    for k, stats in out["summary"]["setupII_tests"].items():
        CI = stats["CI95"]
        print(f"- {k}: r_hat={stats['r_hat']:.4f}, CI=({CI[0]:.4f}, {CI[1]:.4f}), p={stats['p_value']:.4g}")


if __name__ == "__main__":
    main()

