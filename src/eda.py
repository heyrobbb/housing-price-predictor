import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('../data/housing.csv')

print("Data shape:\n", df.shape)
print("Columns:\n", df.columns.tolist())
print("Data Types:\n", df.dtypes)
print("Summary Stats:\n", df.describe())

os.makedirs("plots", exist_ok=True)

correlation = df.corr(numeric_only=True)
plt.figure(figsize=(10,8)) 
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.tight_layout()
plt.savefig('../plots/correlation_heatmap.png')
print('Correlation heatmap saved.')

# Distribution of target 
plt.figure(figsize=(8, 6))
sns.histplot(df['MedHouseVal'], bins=30, kde=True)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('../plots/target_distribution.png')
print('Target distribution plot saved.')
