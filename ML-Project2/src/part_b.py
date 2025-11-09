import torch
import numpy as np

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

def get_model(input_dim, hidden_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),     # Input layer
        torch.nn.ReLU(),                                                                # Activation function
        torch.nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True),    # Output layer, no activation function in the end because we want the target to be a value                                                        # Output activation function (for binary classification)
    )

def ann_model(X,y, k=(10,10), hidden_dims=[1,2,10,50], lr=0.001, n_epochs=1000, seed=1234, show_plot=True):
    
    input_dim = X.shape[1]
    hidden_dims = [1, 2, 10, 50]

    output_dim = 1


    K1 = 10
    K2 = 10
    CV_outer = KFold(K1, shuffle=True, random_state=seed)

    # Define hyperparameters
    lr = 0.001
    n_epochs = 1000

    results = {}
    per_fold_best = []

    for k1, (train_index_outer, val_index_outer) in enumerate(CV_outer.split(X, y)):
        print(f'Fold {k1+1}/{K1}')
        
        X_train_outer, X_val_outer = X[train_index_outer], X[val_index_outer]
        y_train_outer, y_val_outer = y[train_index_outer], y[val_index_outer]

        CV_inner = KFold(n_splits=K2, shuffle=True, random_state=seed)
        inner_scores = {hd: [] for hd in hidden_dims}

        for k2, (train_index_inner, val_index_inner) in enumerate(CV_inner.split(X_train_outer, y_train_outer)):
            print(f'Fold {k2+1}/{K2}')
            
            X_train_inner = X_train_outer[train_index_inner]        
            X_val_inner   = X_train_outer[val_index_inner]               
            y_train_inner = y_train_outer[train_index_inner]              
            y_val_inner   = y_train_outer[val_index_inner]

            mean, std = X_train_inner.mean(axis=0), X_train_inner.std(axis=0)
            std[std == 0.0] = 1.0  # avoid /0
            X_train = (X_train_inner - mean) / std
            X_val   = (X_val_inner   - mean) / std

            # Convert to torch tensors **from the NORMALIZED arrays**
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train_inner, dtype=torch.float32).view(-1, 1)
            X_val   = torch.tensor(X_val, dtype=torch.float32)
            y_val   = torch.tensor(y_val_inner, dtype=torch.float32).view(-1, 1)

            # Set up a dictionary to store the results for each hyperparameter setting
            results_inner = {hidden_dim: {'train': [], 'val': []} for hidden_dim in hidden_dims}

            for hidden_dim in hidden_dims:
                model = get_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                # train on inner-train
                for epoch in range(n_epochs):
                    model.train()
                    optimizer.zero_grad()
                    loss = criterion(model(X_train), y_train)
                    loss.backward()
                    optimizer.step()

                # validate on inner-val
                model.eval()
                with torch.no_grad():
                    val_mse = criterion(model(X_val), y_val).item()
                inner_scores[hidden_dim].append(val_mse)

            print(f'  Inner fold {k2+1}/{K2} done')

        avg_inner = {hd: float(np.mean(mses)) for hd, mses in inner_scores.items()}
        best_hidden = min(avg_inner, key=avg_inner.get)
        print(f'  Selected hidden_dim={best_hidden} (inner-CV mean MSE={avg_inner[best_hidden]:.4f})')


        # Normalize data here
        mean, std = X_train_outer.mean(axis=0), X_train_outer.std(axis=0)
        X_train_outer = (X_train_outer - mean) / std
        X_val_outer = (X_val_outer - mean) / std

        # Convert to torch tensors
        X_train_outer = torch.tensor(X_train_outer, dtype=torch.float32)
        y_train_outer = torch.tensor(y_train_outer, dtype=torch.float32).view(-1, 1)
        X_val_outer = torch.tensor(X_val_outer, dtype=torch.float32)
        y_val_outer = torch.tensor(y_val_outer, dtype=torch.float32).view(-1, 1)

        results_inner = {hidden_dim: {'train': [], 'val': []} for hidden_dim in hidden_dims}
        best_outer_mse = None

        for hidden_dim in hidden_dims:
            model = get_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            criteration = torch.nn.MSELoss()  # keep your name
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
            
            for epoch in range(n_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_outer)
                loss = criteration(outputs, y_train_outer)
                loss.backward()
                optimizer.step()

                results_inner[hidden_dim]['train'].append(loss.item())

            # Outer validation
            with torch.no_grad():
                model.eval()
                val_outputs = model(X_val_outer)
                val_loss = criteration(val_outputs, y_val_outer)
                results_inner[hidden_dim]['val'].append(val_loss.item())
                if hidden_dim == best_hidden:
                    best_outer_mse = val_loss.item()
                print(f'    [Outer eval] Hidden units: {hidden_dim}, Validation MSE: {val_loss.item():.4f}')

        results[k1] = results_inner
        per_fold_best.append({
            "fold": k1+1,
            "best_hidden_dim": best_hidden,
            "outer_val_mse_of_best": best_outer_mse,
            "inner_mean_mse_per_hd": avg_inner
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

    return per_fold_best