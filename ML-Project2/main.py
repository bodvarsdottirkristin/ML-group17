import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.lines import Line2D

from src.part_a import ridge_regression
from src.part_b import ann_model
from src.part_c import *

import pandas as pd
import numpy as np
import scipy.stats as st

def fetch_data():
    """
    Loads and clean the SAheart dataset.
    Input:      None
    Output:     X : np.ndarray (Feature matrix)
                y : np.ndarray (Target vector (ldl values))
                feature_names : list (Feature names)
    """
    # Load data
    df = pd.read_csv('data/SAHeart.csv', index_col=0)

    # Convert 'famhist' to numbers (0 = Absent, 1 = Present)
    df['famhist'] = df['famhist'].map({'Absent': 0, 'Present': 1})

    # Split into features and target
    feature_names = df.drop(columns=['ldl', 'chd']).columns.tolist()
    X = df.drop(columns=['ldl', 'chd']).values
    y_reg = df['ldl'].values

    # Split into features and target for classification
    X_cat = df.drop(columns=['chd']).values
    y_cat = df['chd'].values
    
    return X, y_reg, y_cat, feature_names

def main():

    SEED = 1234

    # Better Grids ?
    lambdas = np.logspace(-4, 5, 20)

    # Fetch data
    X, y_reg, y_cat, feature_names = fetch_data()



    # Regression part A
    ridge_a = ridge_regression(X, y_reg, lambdas=lambdas, seed=SEED, show_plot=True)
    print(ridge_a)
    
    # Print the linear function
    print("\n" + "="*60)
    print("Ridge Regression Linear Function (with optimal lambda)")
    print("="*60)
    print(f"Optimal lambda: {ridge_a['best_lambda']:.6e}")
    print(f"Intercept: {ridge_a['intercept']:.6f}")
    print("\nCoefficients (on standardized features):")
    for name, coef in zip(feature_names, ridge_a['coefficients']):
        print(f"  {name:15s}: {coef:10.6f}")
    
    # Print the linear function equation (standardized features)
    print("\nLinear Function (standardized features):")
    print(f"ldl = {ridge_a['intercept']:.6f}", end="")
    for name, coef in zip(feature_names, ridge_a['coefficients']):
        sign = "+" if coef >= 0 else "-"
        print(f" {sign} {abs(coef):.6f} * {name}_std", end="")
    print("\n")
    print("(Note: Features are standardized before applying coefficients)")
    
    # Convert to original (non-standardized) features
    # y = intercept + sum(beta_i * (x_i - mu_i) / sigma_i)
    #   = intercept - sum(beta_i * mu_i / sigma_i) + sum((beta_i / sigma_i) * x_i)
    mu = ridge_a['scaler_mean']
    sigma = ridge_a['scaler_std']
    original_coefs = ridge_a['coefficients'] / sigma
    original_intercept = ridge_a['intercept'] - np.sum(ridge_a['coefficients'] * mu / sigma)
    
    print("\nLinear Function (original features):")
    print(f"y = {original_intercept:.6f}", end="")
    for name, coef in zip(feature_names, original_coefs):
        sign = "+" if coef >= 0 else "-"
        print(f" {sign} {abs(coef):.6f} * {name}", end="")
    print("\n")
    print("="*60 + "\n")

    # Regression part B
    table1, results, ttest_df  = ann_model(X, y_reg, k=(10,10), hidden_dims=[1,2,3,4,5,10,50], lr=0.001, n_epochs=1000, seed=SEED)
    print(table1)
    print(ttest_df)


    # Part B - 3. Statistical evaluation, setup I
    print(results)
    
    # Classification

    results = run_classification(X, y_cat, feature_names=feature_names, K_outer=10, K_inner=10, seed=1234)
    print(results)



if __name__ == "__main__":
    main()

