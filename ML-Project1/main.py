
# Adding the neccessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load data from csv 
df = pd.read_csv(r'data/online_shoppers_intention.csv')

N = df.shape[0]
M = df.shape[1]

print(f'N: {N}')
print(f'M: {M}')

# The dataset has 12.330 instances and 18 features

# Lets print the columns
print(df.info())

# Split into X (features) and y (target)
X = df.drop(columns=["Revenue"])
y = pd.Categorical(df['Revenue']) # type: ignore[attr-defined]


## 3. 1 - SUMMARY STATISTICS
# I'm copying straight from terminal, so iterate bc it only shows 5 at a time

col_groups = np.array_split(df.columns, 3)
for i, cols in enumerate(col_groups, start=1):
    print(f"\nSummary Table {i}")
    print(df[cols].describe().round(2))

# # Display the first few rows of the dataframe
# print(X.head())