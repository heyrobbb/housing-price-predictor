from sklearn.datasests import fetch_california_housing
import pandas as pd

# load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

df.to_csv('data/housing.csv', index=False)

print(df.head())
