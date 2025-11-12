import numpy as np
from sklearn.pipeline import Pipeline
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import chi2, binomtest
import scipy.stats as st

# ---------- Utilities ----------
def standardize(X_train, X_test):
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma

def split_test_train(X, y, train_idx, test_idx):
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

# ---------- Baseline ----------
def baseline_predict(y_train, n_test):
    # Predict the majority class from TRAINING data for all TEST points
    vals, counts = np.unique(y_train, return_counts=True)
    majority = vals[np.argmax(counts)]
    return np.full(n_test, majority, dtype=int), int(majority)

# ---------- Logistic Regression (Method: LR) ----------
def log_reg_outer(X_train_outer, y_train_outer, X_test_outer, y_test_outer,
                  lambdas, CV_inner, fold_outer_idx, K_inner):

    test_errors_inner = np.empty((K_inner, len(lambdas)))

    # Inner CV to pick lambda
    for fold_inner_idx, (tr_in_idx, te_in_idx) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):
        X_tr_in, y_tr_in, X_te_in, y_te_in = split_test_train(X_train_outer, y_train_outer, tr_in_idx, te_in_idx)
        X_tr_in, X_te_in = standardize(X_tr_in, X_te_in)

        for lam_idx, lam in enumerate(lambdas):
            Cval = 1.0 / lam
            model = LogisticRegression(penalty='l1', C=Cval, solver='liblinear', max_iter=1000)
            model.fit(X_tr_in, y_tr_in)
            yhat = model.predict(X_te_in)
            test_errors_inner[fold_inner_idx, lam_idx] = error_rate(y_te_in, yhat)

    opt_idx = np.argmin(np.mean(test_errors_inner, axis=0))
    opt_lambda = lambdas[opt_idx]

    # Refit on outer-train with optimal lambda and evaluate on outer-test
    X_tr, X_te = standardize(X_train_outer, X_test_outer)
    model = LogisticRegression(penalty='l1', C=1.0/opt_lambda, solver='liblinear', max_iter=1000)
    model.fit(X_tr, y_train_outer)
    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)

    return opt_lambda, error_rate(y_train_outer, yhat_tr), error_rate(y_test_outer, yhat_te), yhat_te

# ---------- ANN (Method 2) ----------
def get_model(input_dim, hidden_dim, output_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim, bias=True),
        torch.nn.Sigmoid()
    )

def ann_outer(X_train_outer, y_train_outer, X_test_outer, y_test_outer,
              hidden_dims, CV_inner, fold_outer_idx, K_inner,
              lr=0.01, n_epochs=500, seed=1234):

    val_losses_inner = np.empty((K_inner, len(hidden_dims)))

    # Inner CV: select hidden_dim minimizing validation error rate (we’ll evaluate by BCE but rank by error)
    for fold_inner_idx, (tr_in_idx, te_in_idx) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):
        X_tr_in, y_tr_in, X_te_in, y_te_in = split_test_train(X_train_outer, y_train_outer, tr_in_idx, te_in_idx)
        X_tr_in, X_te_in = standardize(X_tr_in, X_te_in)

        torch.manual_seed(seed)
        X_tr_in = torch.tensor(X_tr_in, dtype=torch.float32)
        y_tr_in = torch.tensor(y_tr_in.reshape(-1, 1), dtype=torch.float32)
        X_te_in = torch.tensor(X_te_in, dtype=torch.float32)
        y_te_in = torch.tensor(y_te_in.reshape(-1, 1), dtype=torch.float32)

        for h_idx, h in enumerate(hidden_dims):
            model = get_model(X_tr_in.shape[1], h, 1)
            criterion = torch.nn.BCELoss()
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            for _ in range(n_epochs):
                model.train()
                out = model(X_tr_in)
                loss = criterion(out, y_tr_in)
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                model.eval()
                out_val = model(X_te_in)
                preds = (out_val >= 0.5).float()
                err = (preds != y_te_in).float().mean().item()
                val_losses_inner[fold_inner_idx, h_idx] = err

    opt_h_idx = np.argmin(np.mean(val_losses_inner, axis=0))
    opt_hidden = hidden_dims[opt_h_idx]

    # Train on outer-train; evaluate on outer-test
    X_tr, X_te = standardize(X_train_outer, X_test_outer)
    torch.manual_seed(seed)
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_train_outer.reshape(-1, 1), dtype=torch.float32) # var vitlaust
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    model = get_model(X_tr.shape[1], opt_hidden, 1)
    criterion = torch.nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for _ in range(n_epochs):
        model.train()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        model.eval()
        out_te = model(X_te_t).view(-1)       
        yhat_te = (out_te >= 0.5).to(torch.int).tolist()
        yhat_te = np.array(yhat_te, dtype=int) 
        

    return opt_hidden, error_rate(y_test_outer, yhat_te), yhat_te

# confidence_interval_comparison function from Exercise 7 - Part 4
def mcnemar(y_true, yhatA, yhatB, alpha=0.05, model_a_name='A', model_b_name='B'):
    """
    Perform McNemar's test to compare the accuracy of two classifiers.

    Parameters:
    - y_true: array-like, true labels
    - yhatA: array-like, predicted labels by classifier A
    - yhatB: array-like, predicted labels by classifier B
    - alpha: float, significance level (default: 0.05)

    Returns:
    - E_theta: float, estimated difference in accuracy between classifiers A and B (theta_hat)
    - CI: tuple, confidence interval of the estimated difference in accuracy
    - p: float, p-value for the two-sided test of whether classifiers A and B have the same accuracy
    """

    # Set up the contingency table
    nn = np.zeros((2, 2))

    # 2.1) Fill in the contingency table
    # Correctness indicators
    cA = yhatA == y_true
    cB = yhatB == y_true

    # Fill the contingency table
    nn[0, 0] = sum([cA[i] * cB[i] for i in range(len(cA))]) 
    # Or a bit smarter: nn[0, 0] = sum(cA & cB)
    nn[0, 1] = sum(cA & ~cB)
    nn[1, 0] = sum(~cA & cB)
    nn[1, 1] = sum(~cA & ~cB)

    # get values from the contingency table
    n = len(y_true)
    n12 = nn[0, 1]
    n21 = nn[1, 0]

    # 2.2) Calculate E_theta and Q from the values in the contingency table
    E_theta = (n12 - n21) / n

    Q = (
        n**2
        * (n + 1)
        * (E_theta + 1)
        * (1 - E_theta)
        / ((n * (n12 + n21) - (n12 - n21) ** 2))
    )


    # 2.3) Calculate f and g for the beta distribution
    f = (E_theta + 1)/2 * (Q - 1)
    g = (1 - E_theta)/2 * (Q - 1)

    # Calculate confidence interval
    CI = tuple(bound * 2 - 1 for bound in st.beta.interval(1 - alpha, a=f, b=g))

    # Calculate p-value for the two-sided test using exact binomial test
    p = 2 * st.binom.cdf(min([n12, n21]), n=n12 + n21, p=0.5)

    print(f"Result of McNemars test using alpha = {alpha}\n")
    print("Contingency table")
    print(nn, "\n")
    if n12 + n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=", (n12 + n21))

    print(f"Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = {CI[0]:.4f}, {CI[1]:.4f}\n")
    print(
        f"p-value for two-sided test {model_a_name} and {model_b_name} have same accuracy (exact binomial test): p={p}\n"
    )

    return E_theta, CI, p


# ---------- Main runner ----------
def run_classification(X, y, K_outer=10, K_inner=10, seed=1234,
                       lambdas=(1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5),
                       hidden_dims=(1, 2, 5, 10, 20, 50),
                       ann_lr=0.01, ann_epochs=500):

    assert set(np.unique(y)).issubset({0, 1}), "y must be binary {0,1}"

    CV_outer = StratifiedKFold(n_splits=K_outer, shuffle=True, random_state=seed)

    table_rows = []  # For Table 2
    # Collect predictions for statistics
    y_true_all = []
    y_hat_lr_all = []
    y_hat_ann_all = []
    y_hat_base_all = []
    lam_stars = []
    for i, (tr_idx, te_idx) in enumerate(CV_outer.split(X, y), start=1):
        X_tr, y_tr, X_te, y_te = split_test_train(X, y, tr_idx, te_idx)
        CV_inner = StratifiedKFold(n_splits=K_inner, shuffle=True, random_state=seed)

        # Logistic regression
        lam_star, train_err_lr, test_err_lr, yhat_lr = log_reg_outer(
            X_tr, y_tr, X_te, y_te, lambdas, CV_inner, i-1, K_inner
        )

        # Method 2: ANN
        h_star, test_err_ann, yhat_ann = ann_outer(
            X_tr, y_tr, X_te, y_te, hidden_dims, CV_inner, i-1, K_inner,
            lr=ann_lr, n_epochs=ann_epochs, seed=seed
        )

        # Baseline
        yhat_base, base_class = baseline_predict(y_tr, len(y_te))
        test_err_base = error_rate(y_te, yhat_base)

        # Save row: [outer i, Method2 param, Method2 Etest, λ*, LR Etest, Baseline Etest]
        table_rows.append([i, h_star, test_err_ann, lam_star, test_err_lr, test_err_base])

        # For stats
        y_true_all.append(y_te)
        y_hat_lr_all.append(yhat_lr)
        y_hat_ann_all.append(yhat_ann)
        y_hat_base_all.append(yhat_base)
        lam_stars.append(lam_star)

        print(f"Outer fold {i}: ANN(h*={h_star}) E_test={test_err_ann:.3f} | "
              f"LR(λ*={lam_star}) E_test={test_err_lr:.3f} | "
              f"Baseline(most={base_class}) E_test={test_err_base:.3f}")

    table = np.array(table_rows, dtype=object)

    # Concatenate predictions for paired tests
    y_true_cat = np.concatenate(y_true_all)
    y_lr_cat   = np.concatenate(y_hat_lr_all)
    y_ann_cat  = np.concatenate(y_hat_ann_all)
    y_base_cat = np.concatenate(y_hat_base_all)

    # Pairwise McNemar
    e_theta_lr_vs_ann, ci_lr_vs_ann,  p_lr_vs_ann = mcnemar(y_true_cat, y_lr_cat, y_ann_cat, model_a_name='Logistic regression', model_b_name='ANN')
    e_theta_lr_vs_base, ci_lr_vs_base, p_lr_vs_base = mcnemar(y_true_cat, y_lr_cat, y_base_cat, model_a_name='Logistic regression', model_b_name='Baseline')
    e_theta_ann_vs_base, ci_ann_vs_base, p_ann_vs_base = mcnemar(y_true_cat, y_ann_cat, y_base_cat, model_a_name='ANN', model_b_name='Baseline' )

    stats = {
        "LR vs ANN": {
            "e_theta": e_theta_lr_vs_ann, 
            "p_value": p_lr_vs_ann,
            "CI": ci_lr_vs_ann
        },
        "LR vs Baseline": {
            "e_theta": e_theta_lr_vs_base, "p_value": p_lr_vs_base,
            "CI": ci_lr_vs_base
        },
        "ANN vs Baseline": {
            "e_theta": e_theta_ann_vs_base, "p_value": p_ann_vs_base,
            "CI": ci_ann_vs_base
        }
    }

    # Pretty print Table 2–style header
    print("\nTwo-level cross-validation table (error rates in %):")
    print("Outer fold | Method 2 (h*) | E_test(Method2) | λ* (LR) | E_test(LR) | E_test(Baseline)")
    for r in table:
        print(f"{r[0]:>10} | {r[1]:>12} | {r[2]:>14.2f} | {r[3]:>7} | {r[4]:>10.2f} | {r[5]:>15.2f}")
    
    #TODO: Bæta við part 5: Train a logistic regression
    lambda_star_final = float(np.median(lam_star.values))
    C_final = 1.0 / lambda_star_final

    return {
        "table_rows": table,          # array with columns: i, h*, E_ann, λ*, E_lr, E_base (E's in %)
        "mcnemar_stats": stats        # dict of p-values & CIs for three pairwise tests
    }

