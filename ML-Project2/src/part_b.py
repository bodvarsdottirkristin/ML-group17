import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import scipy.stats as st

def standardize_train_test(X_train, X_test):
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma == 0] = 1.0       

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test

def squared_error(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return (y_true - y_pred) ** 2

def get_model(input_dim, hidden_dim, output_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim, bias=True),
    )

# confidence_interval_comparison function from Exercise 7 - Part 4
def confidence_interval_comparison(y_true, y_preds_A, y_preds_B, loss_fn, alpha=0.05):

    # Calculate estimated error, z_hat, as the mean loss across all samples
    z = loss_fn(y_true, y_preds_A) - loss_fn(y_true, y_preds_B)
    z_hat = np.mean(z)
    
    # n and nu
    n = len(y_true)
    nu = n - 1  # degrees of freedom

    sem = np.sqrt(sum(((z - z_hat)**2) / (n * nu))) # or st.sem(loss_fn(y_true, y_preds))
    
    CI = st.t.interval(1 - alpha, df=nu, loc=z_hat, scale=sem)  # Confidence interval

    t_stat = -np.abs(np.mean(z)) / st.sem(z)
    p_value = 2 * st.t.cdf(t_stat, df=nu)  # p-value

    return z_hat, CI, p_value

# CHAT CREATED
def paired_t_on_outer_mse(mse_A, mse_B, alpha=0.05):
    """
    mse_A, mse_B: arrays of length K1 with outer-fold test MSEs for model A and B.
    Returns (mean_diff, (CI_low, CI_high), p_value) where diff = A - B.
    """
    import numpy as np, scipy.stats as st
    d = np.asarray(mse_A) - np.asarray(mse_B)
    K = d.size
    d_bar = d.mean()
    s = d.std(ddof=1)
    sem = s / np.sqrt(K)
    t = d_bar / sem
    p = 2 * st.t.sf(np.abs(t), df=K-1)
    ci = st.t.interval(0.95, df=K-1, loc=d_bar, scale=sem)
    return float(d_bar), (float(ci[0]), float(ci[1])), float(p)
##

def ann_model(X, y, k=(10,10), hidden_dims=[1,2,3,4,5,10,50], lr=0.001, n_epochs=1000, seed=1234, show_plot=False,  lambdas=None):

    if lambdas is None:
        lambdas = np.logspace(-3,3,30)

    def train_ann_once(Xtr_np, ytr_np, h, lr, n_epochs, seed):
        torch.manual_seed(seed)
        X_train_t = torch.tensor(Xtr_np, dtype=torch.float32)
        y_train_t = torch.tensor(ytr_np, dtype=torch.float32).view(-1, 1)
        model = get_model(input_dim=X_train_t.shape[1], hidden_dim=h, output_dim=1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        for _ in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
        return model

    K1, K2 = k
    CV_outer = KFold(n_splits=K1, shuffle=True, random_state=seed)

    rows = []
    ann_inner_history = []

    y_true_all      = []
    y_pred_ann_all  = []
    y_pred_ridge_all= []
    y_pred_base_all = []

    for k1, (train_index_outer, test_index_outer) in enumerate(CV_outer.split(X, y), start=1):
        print(f'Fold {k1}/{K1}')

        # Outer split
        X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
        y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

        # INNER CV for ANN: pick h*
        CV_inner = KFold(n_splits=K2, shuffle=True, random_state=seed)
        inner_splits = list(CV_inner.split(X_train_outer, y_train_outer))
        inner_scores_ann = {h: {"train": [], "val": []} for h in hidden_dims}

        for k2, (train_index_inner, val_index_inner) in enumerate(inner_splits):
            # Inner split
            X_train_inner = X_train_outer[train_index_inner]
            X_val_inner   = X_train_outer[val_index_inner]
            y_train_inner = y_train_outer[train_index_inner]
            y_val_inner   = y_train_outer[val_index_inner]

            X_train_std, X_val_std = standardize_train_test(X_train_inner, X_val_inner)

            # Evaluate all h on inner-val
            for h in hidden_dims:
                model_ann = train_ann_once(X_train_std, y_train_inner, h=h, lr=lr, n_epochs=n_epochs, seed=seed)
                with torch.no_grad():
                    yhat_tr_t = model_ann(torch.tensor(X_train_std, dtype=torch.float32))
                    yhat_va_t = model_ann(torch.tensor(X_val_std,   dtype=torch.float32))
                    yhat_tr = yhat_tr_t.detach().cpu().view(-1).tolist()
                    yhat_va = yhat_va_t.detach().cpu().view(-1).tolist()
                tr_mse = mean_squared_error(y_train_inner, yhat_tr)
                va_mse = mean_squared_error(y_val_inner,   yhat_va)
                inner_scores_ann[h]["train"].append(tr_mse)
                inner_scores_ann[h]["val"].append(va_mse)

        avg_inner_val = {h: float(np.mean(inner_scores_ann[h]["val"])) for h in hidden_dims}
        h_star = min(avg_inner_val, key=avg_inner_val.get)
        print(f'    Selected hidden_dim={h_star} (inner-CV mean MSE={avg_inner_val[h_star]:.4f})')

        mean_train_curve = [float(np.mean(inner_scores_ann[h]["train"])) for h in hidden_dims]
        mean_val_curve   = [float(np.mean(inner_scores_ann[h]["val"]))   for h in hidden_dims]
        ann_inner_history.append({
            "fold": k1,
            "h": list(hidden_dims),
            "mean_train": mean_train_curve,
            "mean_val":   mean_val_curve,
            "h_star": int(h_star)
        })
        # INNER CV for Ridge: pick lambda*
        inner_scores_ridge = []
        for lam in lambdas:
            fold_mse = []
            for train_index_inner, val_index_inner in inner_splits:
                X_train_inner = X_train_outer[train_index_inner]
                X_val_inner   = X_train_outer[val_index_inner]
                y_train_inner = y_train_outer[train_index_inner]
                y_val_inner   = y_train_outer[val_index_inner]

                X_train_std, X_val_std = standardize_train_test(X_train_inner, X_val_inner)

                ridge = Ridge(alpha=float(lam), fit_intercept=True, random_state=seed)
                ridge.fit(X_train_std, y_train_inner)
                yhat = ridge.predict(X_val_std)
                fold_mse.append(mean_squared_error(y_val_inner, yhat))
            inner_scores_ridge.append(np.mean(fold_mse))

        lambda_star = float(lambdas[int(np.argmin(inner_scores_ridge))])
        print(f'    Selected lambda={lambda_star:.3g} (inner-CV mean MSE={np.min(inner_scores_ridge):.4f})')

        # Standardize using outer-train stats only
        X_train_std, X_test_std = standardize_train_test(X_train_outer, X_test_outer)

        # ANN h*
        ann_star = train_ann_once(X_train_std, y_train_outer, h=h_star, lr=lr, n_epochs=n_epochs, seed=seed)

        with torch.no_grad():
            yhat_ann_t = ann_star(torch.tensor(X_test_std, dtype=torch.float32))
            yhat_ann = yhat_ann_t.detach().cpu().view(-1).tolist()
        ann_Etest = float(mean_squared_error(y_test_outer, yhat_ann))
        print(f'    [Outer eval] ANN(h*={h_star}) Test MSE: {ann_Etest:.4f}')

        # Ridge lambda*
        ridge_star = Ridge(alpha=lambda_star, fit_intercept=True, random_state=seed)
        ridge_star.fit(X_train_std, y_train_outer)
        yhat_ridge = ridge_star.predict(X_test_std)
        ridge_Etest = float(mean_squared_error(y_test_outer, yhat_ridge))
        print(f'    [Outer eval] Ridge(λ*={lambda_star:.3g}) Test MSE: {ridge_Etest:.4f}')

        # Baseline
        y_mean = float(np.mean(y_train_outer))
        yhat_base = np.full_like(y_test_outer, y_mean, dtype=float)
        baseline_Etest = float(mean_squared_error(y_test_outer, np.full_like(y_test_outer, y_mean, dtype=float)))
        print(f'    [Outer eval] Baseline Test MSE: {baseline_Etest:.4f}')

        y_true_all.append(y_test_outer.astype(float).ravel())
        y_pred_ann_all.append(np.asarray(yhat_ann, dtype=float).ravel())
        y_pred_ridge_all.append(np.asarray(yhat_ridge, dtype=float).ravel())
        y_pred_base_all.append(np.asarray(yhat_base, dtype=float).ravel())

        rows.append({
            "outer_fold": k1,
            "h_star": int(h_star),
            "ann_Etest": ann_Etest,
            "lambda_star": lambda_star,
            "ridge_Etest": ridge_Etest,
            "baseline_Etest": baseline_Etest
        })

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, rec in enumerate(ann_inner_history):
        ax = axes[i]
        h_vals = rec["h"]
        mean_tr = rec["mean_train"]
        mean_va = rec["mean_val"]
        h_best  = rec["h_star"]
        ax.plot(h_vals, mean_va, marker='o', label='Inner CV mean (val)')
        ax.plot(h_vals, mean_tr, marker='s', linestyle='--', label='Inner CV mean (train)')
        ax.axvline(h_best, color='gray', linestyle=':', label=f'h*={h_best}')
        ax.set_title(f'Outer fold {rec["fold"]}')
        ax.set_xlabel('Hidden units (h)')
        ax.set_ylabel('MSE')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("figures/part_b_ann_h_selection.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

    # Create table for the 2-fold CV of the three models
    table1 = pd.DataFrame(rows, columns=["outer_fold","h_star","ann_Etest","lambda_star","ridge_Etest","baseline_Etest"])

    # CHAT GPT GENERATED ------------------------------
    # --- Paired t-tests on outer-fold MSEs (A - B) ---
    mse_ann   = [r["ann_Etest"]      for r in rows]
    mse_ridge = [r["ridge_Etest"]    for r in rows]
    mse_base  = [r["baseline_Etest"] for r in rows]

    print("\nPaired t-tests on outer-fold MSEs (A - B):")
    ttest_rows = []
    for Aname, A, Bname, B in [
        ("ANN",   mse_ann,   "Ridge",    mse_ridge),
        ("ANN",   mse_ann,   "Baseline", mse_base),
        ("Ridge", mse_ridge, "Baseline", mse_base),
    ]:
        dbar, ci, p = paired_t_on_outer_mse(A, B, alpha=0.05)
        print(f"{Aname} - {Bname}: mean Δ={dbar:.4f}, CI[{ci[0]:.4f}, {ci[1]:.4f}], p={p:.4g}")
        ttest_rows.append({
            "Comparison": f"{Aname} - {Bname}",
            "mean_diff (A-B)": dbar,
            "CI_low": ci[0],
            "CI_high": ci[1],
            "p_value": p
        })
    ttest_df = pd.DataFrame(ttest_rows)


    y_true_all       = np.concatenate(y_true_all, axis=0)
    y_pred_ann_all   = np.concatenate(y_pred_ann_all, axis=0)
    y_pred_ridge_all = np.concatenate(y_pred_ridge_all, axis=0)
    y_pred_base_all  = np.concatenate(y_pred_base_all, axis=0)

    loss_fn = squared_error
    alpha = 0.05

    def summarize_pair(name_A, yA, name_B, yB):
        z_hat, CI, p = confidence_interval_comparison(
            y_true=y_true_all,
            y_preds_A=yA,
            y_preds_B=yB,
            loss_fn=loss_fn,
            alpha=alpha
        )
        # Direction note: z_hat = mean(loss_A - loss_B)
        # So z_hat > 0 => A worse than B (higher loss); z_hat < 0 => A better than B
        return {
            "Comparison": f"{name_A} vs {name_B}",
            "z_hat (mean Δloss)": z_hat,
            "CI_low": CI[0],
            "CI_high": CI[1],
            "p_value": p
        }

    summaries = []
    summaries.append(summarize_pair("Linear regression",   y_pred_ridge_all,   "ANN",    y_pred_ann_all))
    summaries.append(summarize_pair("Linear regression",   y_pred_ridge_all,   "Baseline", y_pred_base_all))
    summaries.append(summarize_pair("ANN", y_pred_ann_all, "Baseline", y_pred_base_all))

    summary_df = pd.DataFrame(summaries)

    print("\nPairwise regression comparisons (loss = squared error):")
    print("z_hat = mean(loss_A - loss_B); z_hat < 0 => A better than B")
    with pd.option_context('display.float_format', '{:,.6f}'.format):
        print(summary_df)

    return table1, summary_df, ttest_df

