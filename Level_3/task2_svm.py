import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.decomposition import PCA
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

print(f"Dataset: {df.shape}  Train: {len(X_train)}  Test: {len(X_test)}")

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
results = {}

print(f"\n{'Kernel':>10}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>8}  {'AUC':>8}")
for kernel in kernels:
    svm = SVC(kernel=kernel, probability=True, random_state=42, C=1.0)
    svm.fit(X_train, y_train)
    yp   = svm.predict(X_test)
    prob = svm.predict_proba(X_test)[:, 1]
    results[kernel] = {
        'model' : svm,
        'acc'   : accuracy_score(y_test, yp),
        'prec'  : precision_score(y_test, yp, zero_division=0),
        'rec'   : recall_score(y_test, yp, zero_division=0),
        'f1'    : f1_score(y_test, yp, zero_division=0),
        'auc'   : roc_auc_score(y_test, prob),
        'y_pred': yp,
        'y_prob': prob
    }
    r = results[kernel]
    print(f"{kernel:>10}  {r['acc']:>10.4f}  {r['prec']:>10.4f}  {r['rec']:>10.4f}  {r['f1']:>8.4f}  {r['auc']:>8.4f}")

best_kernel = max(results, key=lambda k: results[k]['auc'])
best = results[best_kernel]

print(f"\nBest Kernel: {best_kernel.upper()} (AUC={best['auc']:.4f})")
print(f"\n{classification_report(y_test, best['y_pred'], target_names=['No Churn', 'Churn'])}")

cv_scores = cross_val_score(SVC(kernel=best_kernel, random_state=42, C=1.0), X_scaled, y, cv=5)
print(f"5-Fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

svm_lin_2d = SVC(kernel='linear', C=1.0, random_state=42)
svm_rbf_2d = SVC(kernel='rbf',    C=1.0, random_state=42)
svm_lin_2d.fit(X_train_pca, y_train)
svm_rbf_2d.fit(X_train_pca, y_train)

def plot_boundary(ax, svm, X_pca, y, title):
    h = 0.05
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlBu',
               edgecolors='k', linewidth=0.3, s=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("SVM Classification - Churn Prediction", fontsize=14, fontweight='bold')

plot_boundary(axes[0, 0], svm_lin_2d, X_pca, y, "Decision Boundary - Linear (PCA 2D)")
plot_boundary(axes[0, 1], svm_rbf_2d, X_pca, y, "Decision Boundary - RBF (PCA 2D)")

colors_map = {'linear': '#e74c3c', 'rbf': '#3498db', 'poly': '#2ecc71', 'sigmoid': '#f39c12'}
for kernel, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1, 0].plot(fpr, tpr, color=colors_map[kernel], lw=2,
                    label=f"{kernel} (AUC={res['auc']:.3f})")
axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].set_title("ROC Curves - All Kernels")
axes[1, 0].legend(loc='lower right')

cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'], ax=axes[1, 1])
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("Actual")
axes[1, 1].set_title(f"Confusion Matrix (Best: {best_kernel})")

plt.tight_layout()
plt.savefig("outputs/L3_T2_svm_classifier.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to outputs/")
