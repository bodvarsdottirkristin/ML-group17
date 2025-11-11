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

import torch
import numpy as np

# We will classify y

def standardize(X_train, X_test):
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma == 0] = 1

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    return X_train, X_test

def split_test_train(X, y, train_idx, test_idx):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test


def eval_logistic_regression(X_train_outer, y_train_outer, X_val_outer, y_val_outer, lambdas, CV_inner, fold_outer_idx, K=10):
    
    train_errors_inner = np.empty([K, K, len(lambdas)])
    test_errors_inner = np.empty([K, K, len(lambdas)])

    optimal_lambdas = np.empty(K)

    # Loop over inner folds
    for fold_inner_idx, (train_inner_idx, test_inner_idx) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):

        X_train_inner, y_train_inner, X_test_inner, y_test_inner = split_test_train(X_train_outer, y_train_outer, train_inner_idx, test_inner_idx)
        X_train_inner, X_test_inner = standardize(X_train_inner, X_test_inner)

        for lam_idx, lam in enumerate(lambdas):
            model = LogisticRegression(l1=lam)
            model.fit(X_train_inner, y_train_inner)

            train_errors_inner[fold_outer_idx, fold_inner_idx, lam_idx] = np.mean((y_train_inner - model.predict(X_train_inner))**2, axis=0)
            test_errors_inner[fold_outer_idx, fold_inner_idx, lam_idx] = np.mean((y_test_inner - model.predict(X_test_inner))**2, axis=0)
    

    optimal_lambda_idx = np.argmin(np.mean(test_errors_inner[fold_outer_idx], axis=0))
    optimal_lambda = lambdas[optimal_lambda_idx]

    X_train_outer, X_test_outer = standardize(X_train_outer, X_test_outer)

    # Create and fit the model with the optimal lambda on the entire outer training set
    model = Ridge(alpha=optimal_lambdas)
    model.fit(X_train_outer, y_train_outer)

    train_error_outer = np.mean((X_train_outer - model.predict(y_train_outer))**2, axis=0)
    test_error_outer = np.mean((y_test_outer - model.predict(X_test_outer))**2, axis=0)

    return optimal_lambda, train_error_outer, test_error_outer



# BREYTA Í CLASSIFICATION?!
def get_model(input_dim, hidden_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),     # Input layer
        torch.nn.ReLU(),                                                                # Activation function
        torch.nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True),    # Output layer, no activation function in the end because we want the target to be a value   
        torch.nn.Softmax(dim=1),                                                      # Output activation function (for binary classification)
    )
   
    # ATH Hvað er input_dim her
    # Ath hvað er output dim her

    # ATH þarf results = {}
    # ATH þarf per_fold_best = []
def ann_model(X_train_outer, y_train_outer, X_val_outer, y_val_outer, CV_inner, fold_outer_idx: int, K=10, output_dim=1, hidden_dims=[1,2,10,50], lr=0.001, n_epochs=1000, seed=1234, show_plot=True):
#results = {}
#per_fold_best = []

    #rows = []
    #ann_inner_history = []

    inner_scores_ann = {h: {"train": [], "val": []} for h in hidden_dims}
    train_error_inner = np.empty([K, K, len(hidden_dims)])
    val_errors_inner = np.empty([K, K, len(hidden_dims)])

    for fold_inner_idx, (train_index_inner, val_index_inner) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):
        # Inner split
        X_train_inner, y_train_inner, X_val_inner, y_val_inner = split_test_train(X_train_outer, y_train_outer, train_index_inner, val_index_inner)
        X_train_inner, X_val_inner = standardize(X_train_inner, X_val_inner)

        torch.manual_seed(seed)
        X_train_inner = torch.tensor(X_train_inner, dtype=torch.float32)
        y_train_inner = torch.tensor(y_train_inner, dtype=torch.float32).view(-1, 1)
        X_val_inner = torch.tensor(X_val_inner, dtype=torch.float32)
        y_val_inner = torch.tensor(y_val_inner, dtype=torch.float32).view(-1, 1)

        # Set up a dictionary to store the results for each hyperparameter setting
        results_inner = {hidden_dim: {'train': [], 'val': []} for hidden_dim in hidden_dims}

        # Loop over the hyperparameter settings        
        for hidden_dim_idx, hidden_dim in enumerate(hidden_dims):
            # Define a model instance with a specific number of hidden units
            
            model = get_model(input_dim=X_train_inner.shape[1], hidden_dim=hidden_dim, output_dim=output_dim)

            # Define loss criterion
            criterion = torch.nn.MSELoss()

            # Define the optimizer as the Adam optimizer (not needed to know the details)
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=.9)
            
            for epoch in range(n_epochs):
                # Implement the training loop here
                # Set the model to training mode
                model.train()

                # Make a forward pass through the model to compute the outputs
                outputs = model(X_train_inner)
                loss = criterion(outputs, y_train_inner)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store the training loss for this epoch in the dictionary
                results_inner[hidden_dim]['train'].append(loss.item())


            # Compute the final test loss on the test set
            with torch.no_grad(): # No need to compute gradients for the validation set
                model.eval()
                val_outputs = model(X_val_inner)
                val_loss = criterion(val_outputs, y_val_inner)

                val_errors_inner[fold_outer_idx, fold_inner_idx, hidden_dim_idx] = val_loss.item()
                results_inner[hidden_dim]['val'].append(val_loss.item())
                #print(f'  Hidden units: {hidden_dim}, Validation set MSE: {val_loss.item():.4f}')

    # results inner, finna hidden dim þar sem train er minnst
    optimal_hidden_dim_idx = np.argmin(np.mean(val_errors_inner[fold_outer_idx]), axis=0)
    optimal_hidden_dim = hidden_dims[optimal_hidden_dim_idx]

    # Standardize
    X_train_outer, X_val_outer = standardize(X_train_outer, X_val_outer)

    torch.manual_seed(seed)
    X_train_outer = torch.tensor(X_train_outer, dtype=torch.float32)
    y_train_outer = torch.tensor(y_train_outer, dtype=torch.float32).view(-1, 1)

    X_val_outer = torch.tensor(X_val_outer, dtype=torch.float32)
    y_val_outer = torch.tensor(y_val_outer, dtype=torch.float32).view(-1, 1)

    # TRAIN
    model = get_model(input_dim=X_train_outer.shape[1], hidden_dim=optimal_hidden_dim, output_dim=output_dim)
    criterion = torch.nn.MSELoss()

    # Define the optimizer as the Adam optimizer (not needed to know the details)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(n_epochs):
        # Implement the training loop here
        # Set the model to training mode
        model.train()

        # Make a forward pass through the model to compute the outputs
        outputs = model(X_train_outer)
        loss = criterion(outputs, y_train_outer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the training loss for this epoch in the dictionary
        #results_outer[hidden_dim]['train'].append(loss.item())

    # Compute the final test loss on the test set
    with torch.no_grad(): # No need to compute gradients for the validation set
        model.eval()
        val_outputs = model(X_val_outer)
        val_loss = criterion(val_outputs, y_val_outer)

        val_errors_inner[fold_outer_idx, fold_inner_idx, hidden_dim_idx] = val_loss.item()
        results_inner[hidden_dim]['val'].append(val_loss.item())
        #print(f'  Hidden units: {hidden_dim}, Validation set MSE: {val_loss.item():.4f}')


    return optimal_hidden_dim, val_loss.item()

def eval_baseline():
    return 1


def run_classification(X, y, K=10, seed=1234):

    results = []

    CV_outer = KFold(n_splits = K, shuffle=True, random_state=seed)

    for outer_fold_idx, (train_idx, test_idx) in enumerate(CV_outer.split(X, y)):

        # Split dataset
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Logistic regression
        #best_lambda, logreg_train_error, logreg_test_error = eval_log_reg(X_train, y_train, X_test, y_test)

        # ANN model

        CV_inner = KFold(n_splits = K, shuffle=True, random_state=seed)

        ann_model(X_train, y_train, X_test, y_test, CV_inner, outer_fold_idx)


        # results.append({
        #     'fold': outer_fold_idx,
        #     'logreg_lambda': best_lambda,
        #     'logreg_train_error': logreg_train_error,
        #     'logreg_test_error': logreg_test_error,
        #     'ann_params':  best_ann_params,
        #     'ann_test_error': ann_test_error
        # })

#    return results
