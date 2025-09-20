import pandas as pd
import matplotlib.pyplot as plt

# Load data from csv 
df = pd.read_csv(r'online_shoppers_intention.csv')
print('df', df)

# Split into X (features) and y (target)
X = df.drop(columns=["Revenue"])
y = pd.Categorical(df['Revenue']) # type: ignore[attr-defined]
print(X)
print(y)

# Display the first few rows of the dataframe
print(X.head())
