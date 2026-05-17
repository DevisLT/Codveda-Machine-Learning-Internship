import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("datasets/churn_train.csv")
test_df  = pd.read_csv("datasets/churn_test.csv")
df = pd.concat([train_df, test_df], ignore_index=True)
df = df.drop(columns=['State', 'Area code'])

le = LabelEncoder()
df['International plan'] = le.fit_transform(df['International plan'].astype(str))
df['Voice mail plan']    = le.fit_transform(df['Voice mail plan'].astype(str))
df['Churn']              = df['Churn'].astype(int)

X = df.drop(columns=['Churn'])
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

coef_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio' : np.exp(model.coef_[0])
}).sort_values('Odds Ratio', ascending=False)

print("Coefficients and Odds Ratios:")
print(coef_df.to_string(index=False))

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc     = auc(fpr, tpr)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Logistic Regression - Churn Prediction", fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], ax=axes[0])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")

axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc='lower right')

top = coef_df.head(10)
colors = ['#e74c3c' if o > 1 else '#2ecc71' for o in top['Odds Ratio']]
axes[2].barh(top['Feature'], top['Odds Ratio'], color=colors)
axes[2].axvline(1, color='black', linewidth=1, linestyle='--')
axes[2].set_xlabel("Odds Ratio")
axes[2].set_title("Top 10 Features by Odds Ratio")

plt.tight_layout()
plt.savefig("outputs/L2_T1_logistic_regression.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to outputs/")
