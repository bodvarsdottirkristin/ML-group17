import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def ridge_regression(X, y, lambdas, K=10, seed=1234, show_plot=False):

    CV = KFold(n_splits=K, shuffle=True, random_state=seed)
    splits = list(CV.split(X, y))

    # Create a matrix to store results
    train_error = np.empty([len(lambdas), K])
    test_error = np.empty([len(lambdas), K])

    for lam_idx, lam in enumerate(lambdas):
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Standardize using training stats
            mu = np.mean(X_train, axis=0)
            sigma = np.std(X_train, axis=0)
            sigma[sigma == 0] = 1.0         # avoid division by 0

            X_train = (X_train - mu) / sigma
            X_test = (X_test - mu) / sigma

            # Train model
            model = Ridge(alpha = lam, fit_intercept=True)
            model.fit(X_train, y_train)

            # Calculate error
            train_error[lam_idx, fold_idx] = np.mean((y_train - model.predict(X_train)) ** 2)
            test_error[lam_idx, fold_idx] = np.mean((y_test - model.predict(X_test)) ** 2)

    mean_train = np.mean(train_error, axis=1)
    mean_test = np.mean(test_error, axis=1)
    best_idx    = int(np.argmin(mean_test))
    best_lambda = float(lambdas[best_idx])
    best_mse    = float(mean_test[best_idx])

    # Train final model on all data with best lambda
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1.0  # avoid division by 0
    X_scaled = (X - mu) / sigma
    
    final_model = Ridge(alpha=best_lambda, fit_intercept=True)
    final_model.fit(X_scaled, y)

    plt.figure(figsize=(10,6))
    plt.semilogx(lambdas, mean_test, marker='o', label='Test (CV mean)')
    plt.semilogx(lambdas, mean_train, marker='s', linestyle='--', label='Train (CV mean)')

    plt.axvline(best_lambda, color='gray', linestyle=':', label=f'Best λ={best_lambda:.2e}')
    plt.title("Generalization Error vs. Regularization Strength (λ)")
    plt.xlabel("Regularization Strength (λ)")
    plt.ylabel("Mean Squared Error (10-fold CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/part_a.png", dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return {
        'lambdas': lambdas,
        'mean_train': mean_train,
        'mean_test': mean_test,
        'best_lambda': best_lambda,
        'best_mse': best_mse,
        'coefficients': final_model.coef_,
        'intercept': final_model.intercept_,
        'scaler_mean': mu,
        'scaler_std': sigma
    }
