
# Adding the neccessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix

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


### Data visualization

## Check issues with extreme values or outliers in the data

# Only retrieve the continuous features for outlier detection
continuous_cols = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues',
    'SpecialDay'
]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])

fig, ax = plt.subplots(figsize=(10,8))
df_scaled[continuous_cols].boxplot(ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_title('Boxplot of continuous variables')
plt.tight_layout()

# There seem to be approx 2 outliers for ProductRelated_Duration, one for Informational, maybe one for Administrative_Duration.

## How are the individual attributes distributed (e.g. normally distributed)?

fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharey=False)
axes = np.array(axes).flatten()

for i, col in enumerate(continuous_cols):
    data = df[col].values
    axes[i].hist(data, bins=20, color=f"C{i}", edgecolor='black')
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

fig.suptitle("Histograms of Continuous Features", fontsize=16)
plt.tight_layout()


# Many of the features seem to have an exponential distribution

# Administrative -> exp fit

# Administrative_Duration  -> exp fit

# Informational  -> exp fit

# Informational_Duration  -> exp fit

# ProductRelated  -> exp fit

# ProductRelated_Duration  -> exp fit

# PageValues  -> exp fit

## Correlation between attributes

scatter_matrix(df[continuous_cols], figsize=(15, 15), diagonal='hist', alpha=0.5, color='blue')
plt.suptitle("Scatter matrix of continuous features", fontsize=16)
plt.show()

# By quickly looking over the scatter matrix we see that ExitRates and BounceRates has a correlation
