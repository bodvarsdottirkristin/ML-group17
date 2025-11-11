from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# We will classify y

def eval_ann():
    return 1

def eval_baseline():
    return 1


def eval_logistic_regression(X_train_outer, y_train_outer, X_test_outer, y_test_outer, lambdas, CV_inner):
    # Loop over inner folds
    for fold_inner_idx, (train_inner_idx, test_inner_idx) in enumerate(CV_inner.split(X_train, y_train)):
        X_train_inner, y_train_inner = X_train_outer[train_inner_idx], y_train_outer[train_inner_idx]
        X_test_inner, y_test_inner = X_train_outer[test_inner_idx], y_train_outer[test_inner_idx]

        # Standardize using training stats
        mu_inner = np.mean(X_train_inner, axis=0)
        sigma_inner = np.std(X_train_inner, axis=0)
        sigma_inner[sigma_inner == 0] = 1.0         # avoid division by 0

        X_train_inner = (X_train_inner - mu_inner) / sigma_inner
        X_test_inner = (X_test_inner - mu_inner) / sigma_inner

        for lam_idx, lam in enumerate(lambdas):

            model = LogisticRegression(l1=lam)
            model.fit(X_train_inner, y_train_inner)

            train_errors_inner[fold_inner_idx, lambda_idx] = np.mean((y_train_inner - model.predict(X_train_inner))**2, axis=0)
            test_errors_inner[fold_inner_idx, lambda_idx] = np.mean((y_test_inner - model.predict(X_test_inner))**2, axis=0)
    

        optimal_lambda_idx = np.argmin(np.mean(test_errors_inner[outer_fold_idx], axis=0))
        optimal_lambda = lambdas[optimal_hyperparameter_idx]

    mu_outer = np.mean(X_train_outer, axis=0)
    sigma_outer = np.std(X_train_outer, axis=0)
    sigma_outer[sigma_outer == 0] = 1

    X_train_outer = (X_train_outer - mu_outer) / sigma_outer
    X_test_outer = (X_test_outer - mu_outer) / sigma_outer

    # Create and fit the model with the optimal lambda on the entire outer training set
    model = Ridge(alpha=optimal_lambda)
    model.fit(X_train_outer, y_train_outer)

    train_error_outer = np.mean((X_train_outer - model.predict(y_train_outer))**2, axis=0)
    test_error_outer = np.mean((y_test_outer - model.predict(X_test_outer))**2, axis=0)

    return optimal_lambda, train_error_outer, test_error_outer


def run_classification(X, y, K=10):

    results = []

    CV_outer = KFold(n_splits = K, shuffle=True, random_state=seed)

    for outer_fold_idx, (train_idx, test_idx) in enumerate(CV_outer.split(X, y)):

        # Split dataset
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Logistic regression
        best_lambda, logreg_train_error, logreg_test_error = eval_log_reg(X_train, y_train, X_test, y_test)

        # ANN model


        results.append({
            'fold': outer_fold_idx,
            'logreg_lambda': best_lambda,
            'logreg_train_error': logreg_train_error,
            'logreg_test_error': logreg_test_error,
            'ann_params':  best_ann_params,
            'ann_test_error': ann_test_error
        })

    return results
