import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset: {df.shape}  Train: {len(X_train)}  Test: {len(X_test)}")

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
print(f"\nBaseline (100 trees): Accuracy = {accuracy_score(y_test, rf_base.predict(X_test)):.4f}")

param_grid = {
    'n_estimators'     : [50, 100, 200],
    'max_depth'        : [None, 5, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=3, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV F1 : {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred  = best_rf.predict(X_test)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
for feat, imp in feat_imp.head(10).items():
    print(f"  {feat:<30} {imp:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Random Forest - Churn Prediction", fontsize=14, fontweight='bold')

top_n  = feat_imp.head(12)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_n)))[::-1]
axes[0].barh(top_n.index[::-1], top_n.values[::-1], color=colors)
axes[0].set_xlabel("Importance Score")
axes[0].set_title("Top Feature Importances")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'], ax=axes[1])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title("Confusion Matrix")

n_trees = [10, 25, 50, 100, 150, 200]
accs = []
for n in n_trees:
    rf_tmp = RandomForestClassifier(
        n_estimators=n, random_state=42, n_jobs=-1,
        max_depth=grid_search.best_params_['max_depth'],
        min_samples_split=grid_search.best_params_['min_samples_split']
    )
    rf_tmp.fit(X_train, y_train)
    accs.append(accuracy_score(y_test, rf_tmp.predict(X_test)))

axes[2].plot(n_trees, accs, 'go-', markersize=6)
axes[2].set_xlabel("Number of Trees")
axes[2].set_ylabel("Test Accuracy")
axes[2].set_title("n_estimators vs Accuracy")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/L3_T1_random_forest.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to outputs/")
