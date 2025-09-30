# Adding the neccessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
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

print(df['famhist'].head())

# famhist is Absent/Present, lets change that to be categorical where 1 = Present, 0 = Absent
df['famhist'] = pd.Categorical(df['famhist']).codes

# Split into X (features) and y (target)

X = df.drop(columns=["chd"])
y = pd.Categorical(df['chd']).codes

### 4. - EXPLORATORY ANALYSIS

## 4. 1 - SUMMARY STATISTICS

col_groups = np.array_split(df.columns, 3)
for i, cols in enumerate(col_groups, start=1):
    print(f"\nSummary Table {i}")
    print(df[cols].describe().round(2))

## Check issues with extreme values or outliers in the data

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



## 4. 2 - SIMILARITY MEASURES

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

### 5. - PRINCIPAL COMPONENT ANALYSIS

# Create a PCA object and fit to the data
pca = PCA()

pca.fit(X)
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

# The graph shows we need 4 principal components to explain over 90% of the attributes's variance

PC_idxs = [0, 1]  # Indices of the principal components to plot
unique_classes = np.unique(y) # Get unique classes from the target variable

B = pca.transform(X)

# Plot PCA of the data
fig = plt.figure()
plt.title("Heart Disease: PCA")
# Plot the data projected onto the principal components, colored by chd
for is_chd in unique_classes:
    mask = (y == is_chd)
    plt.plot(B[mask, PC_idxs[0]], B[mask, PC_idxs[1]], ".", alpha=0.5)

plt.xlabel(f"PC{PC_idxs[0] + 1}")
plt.ylabel(f"PC{PC_idxs[1] + 1}")

bw = 0.2
r = np.arange(1, X.shape[1] + 1)

fig = plt.figure(figsize=(10, 6))
plt.title("HeartDisease: PCA Component Coefficients")
for i, pc in enumerate(V[:, :4].T):
    plt.bar(r + i * bw, pc, width=bw, label=f"PC{i+1}")
plt.xticks(r + bw, X.columns)
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend()
plt.grid()


# Lets now do the same but with normalized attributes 

# Aftur nema normalized
X_no_binary = X.drop(columns="famhist", axis=1)
X_tilde = (X_no_binary - np.mean(X_no_binary, axis=0)) / np.std(X_no_binary, axis=0)

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
plt.title("Variance explained by principal components, normalized attributes")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()

# choose components to plot: PC2 vs PC3
pcx, pcy = 0, 1                    # zero-based indices

fig = plt.figure()
plt.title("Heart Disease: PCA")

for cls in np.unique(y):
    mask = (y == cls)
    plt.scatter(B[mask, pcx], B[mask, pcy], alpha=0.6, label=str(cls))

plt.xlabel(f"PC{pcx+1}")
plt.ylabel(f"PC{pcy+1}")
plt.legend(title="chd")
plt.grid(True, linestyle=":")

# choose components to plot: PC2 vs PC3
pcx, pcy = 1, 2                     # zero-based indices

fig = plt.figure()
plt.title("Heart Disease: PCA")

for cls in np.unique(y):
    mask = (y == cls)
    plt.scatter(B[mask, pcx], B[mask, pcy], alpha=0.6, label=str(cls))

plt.xlabel(f"PC{pcx+1}")
plt.ylabel(f"PC{pcy+1}")
plt.legend(title="chd")
plt.grid(True, linestyle=":")

# choose components to plot: PC2 vs PC3
pcx, pcy = 0, 2                     # zero-based indices

fig = plt.figure()
plt.title("Heart Disease: PCA")

for cls in np.unique(y):
    mask = (y == cls)
    plt.scatter(B[mask, pcx], B[mask, pcy], alpha=0.6, label=str(cls))

plt.xlabel(f"PC{pcx+1}")
plt.ylabel(f"PC{pcy+1}")
plt.legend(title="chd")
plt.grid(True, linestyle=":")


bw = 0.2
r = np.arange(1, X.shape[1] + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("HeartDisease: PCA Component Coefficients (Modified)")

# Use a colormap for better color variety
colors = plt.cm.Set2.colors  

for i, pc in enumerate(V[:, :7].T):
    ax.bar(r + i * bw, pc, width=bw, label=f"PC{i+1}", color=colors[i % len(colors)], alpha=0.85, edgecolor="black")

ax.set_xticks(r + bw * 3)  # center the labels
ax.set_xticklabels(X.columns, rotation=30, ha="right")
ax.set_xlabel("Attributes")
ax.set_ylabel("Component coefficients")

# Move legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()