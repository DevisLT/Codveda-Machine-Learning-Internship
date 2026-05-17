import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("datasets/iris.csv")
print(f"Dataset: {df.shape} | Classes: {df['species'].unique().tolist()}")

X = df.drop(columns=['species'])
le = LabelEncoder()
y = le.fit_transform(df['species'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

k_values  = range(1, 21)
train_acc = []
test_acc  = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, knn.predict(X_train)))
    test_acc.append(accuracy_score(y_test,  knn.predict(X_test)))

best_k = k_values[np.argmax(test_acc)]

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

print(f"\nBest K: {best_k}")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

print("K Value Comparison:")
for k, tr, te in zip(k_values, train_acc, test_acc):
    mark = " <- best" if k == best_k else ""
    print(f"  K={k:2d}  Train={tr:.4f}  Test={te:.4f}{mark}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("KNN Classifier - Iris", fontsize=14, fontweight='bold')

axes[0].plot(k_values, train_acc, 'b-o', markersize=5, label='Train')
axes[0].plot(k_values, test_acc,  'r-o', markersize=5, label='Test')
axes[0].axvline(best_k, linestyle='--', color='green', alpha=0.7, label=f'Best K={best_k}')
axes[0].set_xlabel("K Value")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("K Value vs Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title(f"Confusion Matrix (K={best_k})")

colors = ['#e74c3c', '#2ecc71', '#3498db']
for i, (cls, col) in enumerate(zip(le.classes_, colors)):
    mask = (le.transform(df['species']) == i)
    axes[2].scatter(df['petal_length'][mask], df['petal_width'][mask],
                    color=col, label=cls, alpha=0.7, edgecolors='white')
axes[2].set_xlabel("Petal Length")
axes[2].set_ylabel("Petal Width")
axes[2].set_title("Feature Space (Petal Dimensions)")
axes[2].legend()

plt.tight_layout()
plt.savefig("outputs/L1_T3_knn_classifier.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to outputs/")
