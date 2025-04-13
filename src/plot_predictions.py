import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../output/housing_with_predictions.csv')

plt.figure(figsize=(8,6))
plt.scatter(df['MedHouseVal'], df['PredictedHouseVal'], alpha=0.3)
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Predicted vs Actual')
plt.grid(True)
plt.tight_layout()
plt.savefig('../plots/predicted_vs_actual.png')
print("Saved predicted vs actual scatter plot.")
