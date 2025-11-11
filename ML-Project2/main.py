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
#from src.part_c_test_hm import run_classification
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
    """
    # Load data
    url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data"
    df = pd.read_csv(url, index_col=0)

    # Convert 'famhist' to numbers (0 = Absent, 1 = Present)
    df['famhist'] = df['famhist'].map({'Absent': 0, 'Present': 1})

    # Split into features and target for regression
    X_reg = df.drop(columns=['ldl', 'chd']).values
    y_reg = df['ldl'].values

    # Split into features and target for classification
    X_cat = df.drop(columns=['chd']).values
    y_cat = df['chd'].values
    
    return X_reg, X_cat, y_reg, y_cat

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


def main():

    SEED = 1234
    lambdas = np.logspace(-5, 8, 14)
    # Fetch data
    X_reg, X_cat, y_reg, y_cat = fetch_data()

    # Regression part A
    ridge_a = ridge_regression(X_reg, y_reg, seed=SEED)
    print(ridge_a)

    # Regression part B
    table1  = ann_model(X_reg, y_reg, k=(10,10), hidden_dims=[1,2,3,4,5,10,50], lr=0.001, n_epochs=1000, seed=SEED, show_plot=True)
    print(table1)


    # Part B - 3. Statistical evaluation, setup I
    alpha = 0.05

    # z_hat, CI, p_value = confidence_interval_comparison(y_true, y_preds["Model_A"], y_preds["Model_B"], l2_loss, alpha=alpha)
    print(f"Difference in loss between ANN and linear regression: \nz_hat: {z_hat:.4f}, \nCI: [{CI[0]:.4f}, {CI[1]:.4f}], \np-value: {p_value}")

    # z_hat, CI, p_value = confidence_interval_comparison(y_true, y_preds["Model_A"], y_preds["Model_B"], l2_loss, alpha=alpha)
    print(f"Difference in loss between ANN and baseline: \nz_hat: {z_hat:.4f}, \nCI: [{CI[0]:.4f}, {CI[1]:.4f}], \np-value: {p_value}")



    # Classification

    run_classification(X_cat, y_cat, K=10, seed=1234)

    #ridge_regression(X, y_reg, seed=SEED)



if __name__ == "__main__":
    main()

