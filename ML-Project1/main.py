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

# Load data from csv 
df = pd.read_csv(r'data/SAHeart.csv')

N = df.shape[0]
M = df.shape[1]

print(f'N: {N}')
print(f'M: {M}')

# The dataset has 12.330 instances and 18 features

# Lets print the columns
print(df.info())

print(df['famhist'].head())

# famhist is Absent/Present, lets change that to be categorical where 1 = Present, 0 = Absent
df['famhist'] = df['famhist'].astype('category')
df['chd'] = df['chd'].astype('category')

# Split into X (features) and y (target)

X = df.drop(columns=["chd"])
y = pd.Categorical(df['chd']).codes

### 2 - Attributes of the data

## 2.3 - Summary statistics

col_groups = np.array_split(df.columns, 3)
for i, cols in enumerate(col_groups, start=1):
    print(f"\nSummary Table {i}")
    print(df[cols].describe().round(2))

### 3. - Data visualization

## 3.1 - Exploratory analysis


### 3.1.1 -  Check issues with extreme values or outliers in the data

# Only retrieve the non-nominal features for outlier detection
non_nominal_cols = df.drop(columns=["row.names", "famhist", "chd"]).columns

fig, axes = plt.subplots(1, 2, figsize=(10,8))
# Create a barplot for the two categorical attributes
for ax, col in zip(axes, ['famhist', 'chd']):
    df[col].value_counts().plot(kind='bar', figsize=(6,4), color='skyblue', edgecolor='black', ax=ax)
    ax.set_title(f'Barplot of {col}')
    ax.set_ylabel('Count')
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



### 3.1.2 -  How are the individual attributes distributed (e.g. normally distributed)?

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

fig, axes = plt.subplots(2, 4, figsize=(18,8))
axes = np.array(axes).flatten()


# Fit a distribution on the histograms
for i, col in enumerate(non_nominal_cols):
    data = df[col].values
    sns.histplot(data, ax=axes[i], kde=True, bins=20, color=f"C{i}")
    axes[i].set_title(f"{col} Distribution", fontsize=12)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
fig.suptitle("Histograms of non nominal attributes, with fit", fontsize=16)
plt.tight_layout()



### 3.1.3 - Correlations between attributes

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

## 3.2. - Principal Component Analysis (PCA)

# Start by normalizing the attributes since they are of different range
X_no_binary = X.drop(columns="famhist", axis=1)
X_tilde = (X_no_binary - np.mean(X_no_binary, axis=0)) / np.std(X_no_binary, axis=0)

# Create a PCA object and fit to the data
pca = PCA()

pca.fit(X_tilde)
V = pca.components_.T
# Compute fraction of variance explained
rho = pca.explained_variance_ratio_

# 90% threshold for variance explained
threshold = 0.9
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()


# Histogram of component coefficients for each PC
bw = 0.1
r = np.arange(1, X_tilde.shape[1] + 1)

fig = plt.figure(figsize=(20, 15))
plt.title("HeartDisease: PCA Component Coefficients")
for i, pc in enumerate(V[:, :6].T):
    plt.bar(r + i * bw, pc, width=bw, label=f"PC{i+1}")
plt.xticks(r + bw, X_tilde.columns)
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend()
plt.grid()

# Project X to the PCA space
B = pca.transform(X_tilde)


PC_idxs = [0, 1, 2, 3, 4, 5]

pc_cols = [f"PC{i+1}" for i in range(6)]
B_df = pd.DataFrame(B[:, :6], columns=pc_cols)
B_df["chd"] = y.codes if hasattr(y, "codes") else y  # numeric codes for color

handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=plt.cm.coolwarm(0), markersize=8, label="0 = Absence"),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=plt.cm.coolwarm(255), markersize=8, label="1 = Presence")
]
# Scatter matrix
scatter_matrix(
    B_df[pc_cols],
    figsize=(15, 15),
    diagonal="hist",
    alpha=0.6,
    c=B_df["chd"],        # color by chd
    cmap="coolwarm"
)
plt.figlegend(handles=handles, loc="upper right", bbox_to_anchor=(0.92, 0.92))
plt.suptitle("Scatter Matrix of First 6 Principal Components", fontsize=16)
plt.show()