import pandas as pd
import joblib
import os

# load model (joblib to serial models and load models) 
model = joblib.load('../models/linear_regression_model.pkl')

# load dataset
df = pd.read_csv('../data/housing.csv')

# separate featuress (same as training) 
X = df.drop('MedHouseVal', axis=1)

# predict
predictions = model.predict(X)

df['PredictedHouseVal'] = predictions

# Save results
os.makedirs('output', exist_ok=True)
df.to_csv('../output/housing_with_predictions.csv', index=False)
print('predictions saved to output/housing_with_predictions.csv')
