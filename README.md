# Codveda ML Internship — Tasks Level 1 to 3

Machine Learning tasks completed during internship at Codveda Technologies.
```

## Results Summary

| Level | Task | Model | Dataset | Result |
|-------|------|-------|---------|--------|
| 1 | Data Preprocessing | pandas / sklearn | Churn | Clean pipeline, 80/20 split |
| 1 | Linear Regression | LinearRegression | House Prices | R² = 0.67 |
| 1 | KNN Classifier | KNeighborsClassifier | Iris | Accuracy = 96.7% |
| 2 | Logistic Regression | LogisticRegression | Churn | ROC AUC = 0.83 |
| 2 | Decision Tree | DecisionTreeClassifier | Iris | Accuracy = 96.7% |
| 2 | K-Means Clustering | KMeans | Iris | ARI = 0.62 |
| 3 | Random Forest | RandomForestClassifier | Churn | Accuracy = 95.5%, F1 = 0.82 |
| 3 | SVM | SVC (RBF kernel) | Churn | ROC AUC = 0.92 |
| 3 | Neural Network | TensorFlow / Keras | Iris | Accuracy = 96.7% |

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow
```

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

## How to Run

Run each script from the project root directory:

```bash
python Level_1/task1_preprocessing.py
python Level_1/task2_linear_regression.py
python Level_1/task3_knn.py

python Level_2/task1_logistic_regression.py
python Level_2/task2_decision_tree.py
python Level_2/task3_kmeans.py

python Level_3/task1_random_forest.py
python Level_3/task2_svm.py
python Level_3/task3_neural_network.py
```

Output charts are saved to the `outputs/` folder.

## Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn
