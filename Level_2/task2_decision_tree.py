import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("datasets/iris.csv")
print(f"Dataset: {df.shape}")

X = df.drop(columns=['species'])
le = LabelEncoder()
y = le.fit_transform(df['species'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

print(f"\nFull Tree  -> Depth: {dt_full.get_depth()}  Leaves: {dt_full.get_n_leaves()}")
print(f"Accuracy: {accuracy_score(y_test, dt_full.predict(X_test)):.4f}")

print("\nPruning - max_depth comparison:")
print(f"{'Depth':>6}  {'Train Acc':>10}  {'Test Acc':>10}  {'F1':>8}")

best_depth, best_acc = None, 0
depth_results = {}

for depth in range(1, 11):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    tr = accuracy_score(y_train, dt.predict(X_train))
    te = accuracy_score(y_test,  dt.predict(X_test))
    f1 = f1_score(y_test, dt.predict(X_test), average='weighted')
    depth_results[depth] = (tr, te, f1)
    if te > best_acc:
        best_acc, best_depth = te, depth
    print(f"{depth:>6}  {tr:>10.4f}  {te:>10.4f}  {f1:>8.4f}")

print(f"\nBest max_depth = {best_depth}")

dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred = dt_pruned.predict(X_test)

print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

cv_scores = cross_val_score(dt_pruned, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

print("\nDecision Tree Rules:")
print(export_text(dt_pruned, feature_names=list(X.columns)))

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Decision Tree - Iris", fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(2, 3, (1, 3))
plot_tree(dt_pruned, feature_names=list(X.columns),
          class_names=le.classes_, filled=True, rounded=True, fontsize=10, ax=ax1)
ax1.set_title(f"Decision Tree Structure (max_depth={best_depth})")

ax2 = fig.add_subplot(2, 3, 4)
depths   = list(depth_results.keys())
tr_accs  = [depth_results[d][0] for d in depths]
te_accs  = [depth_results[d][1] for d in depths]
ax2.plot(depths, tr_accs, 'b-o', markersize=5, label='Train')
ax2.plot(depths, te_accs, 'r-o', markersize=5, label='Test')
ax2.axvline(best_depth, linestyle='--', color='green', alpha=0.7)
ax2.set_xlabel("Max Depth")
ax2.set_ylabel("Accuracy")
ax2.set_title("Depth vs Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(2, 3, 5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax3)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.set_title("Confusion Matrix")

ax4 = fig.add_subplot(2, 3, 6)
feat_imp = pd.Series(dt_pruned.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.plot(kind='barh', color='steelblue', ax=ax4)
ax4.set_xlabel("Importance")
ax4.set_title("Feature Importances")

plt.tight_layout()
plt.savefig("outputs/L2_T2_decision_tree.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to outputs/")
