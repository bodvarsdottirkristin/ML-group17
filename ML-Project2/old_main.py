# Group 17
# Authors: 
# Helga María Magnúsdóttir
# Kristín Böðvarsdóttir
# Þorsteinn Björn Guðmundsson 

# Adding the neccessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.lines import Line2D

# Load data
url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data"
df = pd.read_csv(url, index_col=0)


N = df.shape[0]
M = df.shape[1]

print(f'N: {N}')
print(f'M: {M}')

# The dataset has 12.330 instances and 18 features

# Lets print the columns
print(df.info())

print(df['famhist'].head())

# famhist is Absent/Present, lets change that to be categorical where 1 = Present, 0 = Absent
df['famhist'] = pd.Categorical(df['famhist'])
df['chd'] = pd.Categorical(df['chd'])

# Split into X (features) and y (target)

X = df.drop(columns=["chd"])
y = pd.Categorical(df['chd'])

# Only retrieve the non-nominal features for outlier detection
non_nominal_cols = df.drop(columns=[ "famhist", "chd"]).columns

X_non_nominal = X.drop(columns=["famhist"])

print(X_non_nominal)
X_standardized = (X_non_nominal - X_non_nominal.mean()) / X_non_nominal.std()
print(X_standardized.head())

# Make sure we have mean 0 and std 1
print(X_standardized.mean().round(2))
print(X_standardized.std().round(2))