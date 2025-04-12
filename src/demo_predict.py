import joblib
import numpy as np

model = joblib.load('../models/linear_regression_model.pkl') 

sample_input = np.array([[8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]])
predicted_value = model.predict(sample_input)

print(f'prediced median house value: {predicted_value[0]:.4f}')
