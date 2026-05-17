import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
             'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

df = pd.read_csv("datasets/house_prices.csv", sep=r'\s+', header=None, names=col_names)

print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")

X = df.drop(columns=['MEDV'])
y = df['MEDV']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients:")
print(coef_df.to_string(index=False))
print(f"\nIntercept: {model.intercept_:.4f}")

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

print(f"\nTrain R2:  {r2_score(y_train, y_pred_train):.4f}")
print(f"Test  R2:  {r2_score(y_test,  y_pred_test):.4f}")
print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"Test  MSE: {mean_squared_error(y_test,  y_pred_test):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Linear Regression - House Prices", fontsize=14, fontweight='bold')

axes[0].scatter(y_test, y_pred_test, alpha=0.6, color='steelblue', edgecolors='white', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel("Actual Price ($1000s)")
axes[0].set_ylabel("Predicted Price ($1000s)")
axes[0].set_title(f"Actual vs Predicted (R²={r2_score(y_test, y_pred_test):.4f})")

residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.6, color='coral', edgecolors='white', linewidth=0.5)
axes[1].axhline(0, color='black', linestyle='--', lw=1.5)
axes[1].set_xlabel("Predicted Price")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")

colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
axes[2].axvline(0, color='black', linewidth=0.8)
axes[2].set_xlabel("Coefficient Value")
axes[2].set_title("Feature Coefficients")

plt.tight_layout()
plt.savefig("outputs/L1_T2_linear_regression.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to outputs/")
