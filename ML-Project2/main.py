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
from src.part_c_test_hm import run_classification

import pandas as pd
import numpy as np

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

def main():

    SEED = 1234
    lambdas = np.logspace(-5, 8, 14)
    # Fetch data
    X_reg, X_cat, y_reg, y_cat = fetch_data()

    # Regression part A
    #ridge_a = ridge_regression(X, y_reg, seed=SEED)
    #print(ridge_a)

    # Regression part B
    #table1  = ann_model(X, y_reg, k=(10,10), hidden_dims=[1,2,3,4,5,10,50], lr=0.001, n_epochs=1000, seed=SEED, show_plot=True)
    #print(table1)

    # Classification

    run_classification(X_cat, y_cat, K=10, seed=1234)
    #ridge_regression(X, y_reg, seed=SEED)



if __name__ == "__main__":
    main()

