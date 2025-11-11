import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def standardize_train_test(X_train, X_test):
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma == 0] = 1.0       

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test

def get_model(input_dim, hidden_dim, output_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim, bias=True),
    )

def setup_storage_for_experiment(K_outer, K_inner, num_hyperparams):
    # Setup storage for the optimal hyperparameters found from the inner CV
    optimal_hyperparameters = np.empty(K_outer)

    # Setup storage for model coefficients and errors for each experiment in all inner folds
    ws_inner = np.empty((M + 1, K_outer, K_inner, num_hyperparams))
    train_errors_inner = np.empty((K_outer, K_inner, num_hyperparams))
    test_errors_inner = np.empty((K_outer, K_inner, num_hyperparams))

    # Setup storage for model coefficients for each experiment in all outer folds
    ws_outer = {
        'not regularized': np.empty((M + 1, K_outer)),
        'regularized': np.empty((M + 1, K_outer))
    }
    # Setup storage for errors as a dictionary
    errors_outer = {
        'train': {
            'baseline (no features)': np.empty((K_outer, 1)), 
            'not regularized': np.empty((K_outer, 1)),
            'regularized': np.empty((K_outer, 1))
        },
        'test': {
            'baseline (no features)': np.empty((K_outer, 1)), 
            'not regularized': np.empty((K_outer, 1)),
            'regularized': np.empty((K_outer, 1))
        }
    }
    return optimal_hyperparameters, ws_inner, train_errors_inner, test_errors_inner, ws_outer, errors_outer

def ann_model(X, y, k=(10,10), hidden_dims=[1,2,3,4,5,10,50], lr=0.001, n_epochs=1000, seed=1234, show_plot=False):

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

    # Grids
    lambdas = np.logspace(-3, 3, 30)

    K1, K2 = k
    CV_outer = KFold(n_splits=K1, shuffle=True, random_state=seed)

    rows = []
    results = {}

    for k1, (train_index_outer, test_index_outer) in enumerate(CV_outer.split(X, y), start=1):
        print(f'Fold {k1}/{K1}')

        # Outer split
        X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
        y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

        # INNER CV for ANN: pick h*
        CV_inner = KFold(n_splits=K2, shuffle=True, random_state=seed)
        inner_scores_ann = {h: [] for h in hidden_dims}

        for k2, (train_index_inner, val_index_inner) in enumerate(CV_inner.split(X_train_outer, y_train_outer), start=1):
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
                    yhat_t = model_ann(torch.tensor(X_val_std, dtype=torch.float32))
                    yhat = yhat_t.detach().cpu().view(-1).tolist()
                mse = mean_squared_error(y_val_inner, yhat)
                inner_scores_ann[h].append(mse)

        h_star = min(inner_scores_ann, key=lambda h: np.mean(inner_scores_ann[h]))
        print(f'    Selected hidden_dim={h_star} (inner-CV mean MSE={np.mean(inner_scores_ann[h_star]):.4f})')

        # INNER CV for Ridge: pick lambda*
        inner_scores_ridge = []
        for lam in lambdas:
            fold_mse = []
            for train_index_inner, val_index_inner in CV_inner.split(X_train_outer, y_train_outer):
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
        baseline_Etest = float(mean_squared_error(y_test_outer, np.full_like(y_test_outer, y_mean, dtype=float)))
        print(f'    [Outer eval] Baseline Test MSE: {baseline_Etest:.4f}')

        rows.append({
            "outer_fold": k1,
            "h_star": int(h_star),
            "ann_Etest": ann_Etest,
            "lambda_star": lambda_star,
            "ridge_Etest": ridge_Etest,
            "baseline_Etest": baseline_Etest
        })

    fig, axs = plt.subplots(4, 3, figsize=(15, 12), sharey=True, sharex=True)
    axs = axs.ravel()

    for fold in range(K1):
        ax = axs[fold]
        for hidden_dim in hidden_dims:
            ax.plot(results[fold][hidden_dim]['train'], label=f'hidden_dim={hidden_dim}')
        ax.set_title(f'Fold {fold+1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')

    plt.suptitle('Training loss for different hidden units')
    plt.tight_layout()
    axs[0].legend()
    plt.savefig("figures/part_b.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

    # Create table for the 2-fold CV of the three models
    table1 = pd.DataFrame(rows, columns=["outer_fold","h_star","ann_Etest","lambda_star","ridge_Etest","baseline_Etest"])

    return table1

