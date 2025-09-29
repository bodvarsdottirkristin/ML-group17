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
y = pd.Categorical(df['chd']) 

print(df['famhist'].head())
# famhist is Absent/Present, lets change that to be categorical where 1 = Present, 0 = Absent
df['famhist'] = pd.Categorical(df['famhist']).codes

## 4. 1 - SUMMARY STATISTICS

col_groups = np.array_split(df.columns, 3)
for i, cols in enumerate(col_groups, start=1):
    print(f"\nSummary Table {i}")
    print(df[cols].describe().round(2))

## Check issues with extreme values or outliers in the data

# Only retrieve the non-nominal features for outlier detection
non_nominal_cols = df.drop(columns=["row.names", "famhist", "chd"]).columns

# Create a barplot for the two categorical attributes
for i, col in enumerate(['famhist', 'chd']):
    plt.figure(figsize=(6, 4)) 
    df[col].value_counts().plot(kind='bar', figsize=(6,4), color='skyblue', edgecolor='black')
    plt.title(f'Barplot of {col}')
    plt.ylabel('Count')
    plt.tight_layout()

# Create boxplots for the non-nominal attributes, first not normalized
fig, ax = plt.subplots(figsize=(10,8))
df[non_nominal_cols].boxplot(ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_title('Boxplot of non nominal attributes, not normalized')
plt.tight_layout()

# Normalize variables to visualize better
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[non_nominal_cols] = scaler.fit_transform(df[non_nominal_cols])

# Create boxplots for the non-nominal attributes, now normalized
fig, ax = plt.subplots(figsize=(10,8))
df_scaled[non_nominal_cols].boxplot(ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_title('Boxplot of non nominal attributes, normalized')
plt.tight_layout()

## How are the individual attributes distributed (e.g. normally distributed)?

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
axes = np.array(axes).flatten()

for i, col in enumerate(non_nominal_cols):
    data = df[col].values
    axes[i].hist(data, bins=20, color=f"C{i}", edgecolor='black')
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

fig.suptitle("Histograms of non nominal attributes", fontsize=16)
plt.tight_layout()

fig, axes = plt.subplots(2, 4, figsize=(8,12))
axes = np.array(axes).flatten()


# Fit a distribution on the histograms
for i, col in enumerate(non_nominal_cols):
    data = df[col].values
    sns.histplot(data, ax=axes[i], kde=True, bins=20, color=f"C{i}")
    axes[i].set_title(f"{col} Distribution", fontsize=12)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
plt.tight_layout()



## 4. 2 - SIMILATIRY MEASURES

corr = df[non_nominal_cols].corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,       
    fmt=".2f",        
    cmap="mako",  
    cbar=True

)
plt.title("Correlation Heatmap of non nominal attributes", fontsize=16)
plt.tight_layout()

## Scatter matrix : Correlation between attributes
scatter_matrix(df[non_nominal_cols], figsize=(15, 15), diagonal='hist', alpha=0.5, color='blue')
plt.suptitle("Scatter matrix of non nominal attributes", fontsize=16)
plt.show()