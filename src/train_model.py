import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# load dataset
df = pd.read_csv('../data/housing.csv')

# split features and target

# supervised problem 
# Capital X is used (convention) input data (your features) Rows + colums, its capital because it is usually a whole matrix
X = df.drop("MedHouseVal", axis=1)

# lowercase is because it is one column all data is X, but we want to predict y 
y = df["MedHouseVal"]

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MSE: {mse: .4f}')
print('r2 score: {r2: .4f}')

os.makedirs('models', exist_ok=True)
joblib.dump(model, '../models/linear_regression_model.pkl')
print('model saved to ../models/linear_regression_model.pkl')
