
# Adding the neccessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
import seaborn as sns

# Load data from csv 
df = pd.read_csv(r'data/SAHeart.csv')

N = df.shape[0]
M = df.shape[1]

print(f'N: {N}')
print(f'M: {M}')

# The dataset has 12.330 instances and 18 features

# Lets print the columns
print(df.info())

# Split into X (features) and y (target)
X = df.drop(columns=["chd"])
y = pd.Categorical(df['chd']) # type: ignore[attr-defined]


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
df['famhist'] = pd.Categorical(df['famhist']).codes
continuous_cols = df.drop(columns=["row.names", "famhist", "chd"]).columns

for i, col in enumerate(['famhist', 'chd']):
    plt.figure(figsize=(6, 4)) 
    df[col].value_counts().plot(kind='bar', figsize=(6,4), color='skyblue', edgecolor='black')
    plt.title(f'Barplot of {col}')
    plt.ylabel('Count')
    plt.tight_layout()

for col in continuous_cols:
    plt.figure(figsize=(6, 4))  # new figure for each feature
    df.boxplot(column=col)      # or: plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
    plt.tight_layout()

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])


fig, ax = plt.subplots(figsize=(10,8))


df[continuous_cols].boxplot(ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_title('Boxplot of non nominal attributes, not normalized')
plt.tight_layout()

fig, ax = plt.subplots(figsize=(10,8))

df_scaled[continuous_cols].boxplot(ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_title('Boxplot of non nominal attributes, normalized')
plt.tight_layout()

## How are the individual attributes distributed (e.g. normally distributed)?

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
axes = np.array(axes).flatten()

for i, col in enumerate(continuous_cols):
    data = df[col].values
    axes[i].hist(data, bins=20, color=f"C{i}", edgecolor='black')
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

fig.suptitle("Histograms of non nominal attributes", fontsize=16)
plt.tight_layout()

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8,12))
axes = np.array(axes).flatten()
for i, col in enumerate(continuous_cols):
    data = df[col].values
    sns.histplot(data, ax=axes[i], kde=True, bins=20, color=f"C{i}")
    axes[i].set_title(f"{col} Distribution", fontsize=12)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

plt.tight_layout()


corr = df[continuous_cols].corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,       # display correlation coefficients
    fmt=".2f",        # format numbers to 2 decimal places
    cmap="coolwarm",  # color map (blue→white→red)
    cbar=True
)
plt.title("Correlation Heatmap of Continuous Features", fontsize=16)
plt.tight_layout()

## Scatter matrix : Correlation between attributes

scatter_matrix(df[continuous_cols], figsize=(15, 15), diagonal='hist', alpha=0.5, color='blue')
plt.suptitle("Scatter matrix of continuous features", fontsize=16)
plt.show()