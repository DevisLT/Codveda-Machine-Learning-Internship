import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("datasets/iris.csv")
print(f"Dataset: {df.shape}  Classes: {df['species'].unique().tolist()}")
print(f"TensorFlow: {tf.__version__}")

X = df.drop(columns=['species']).values
le = LabelEncoder()
y_int = le.fit_transform(df['species'])
y = keras.utils.to_categorical(y_int, num_classes=3)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, yi_train, yi_test = train_test_split(
    X_scaled, y, y_int, test_size=0.2, random_state=42, stratify=y_int
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

tf.random.set_seed(42)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,), name='Hidden_1'),
    layers.Dropout(0.2,                                    name='Dropout_1'),
    layers.Dense(32, activation='relu',                    name='Hidden_2'),
    layers.Dropout(0.2,                                    name='Dropout_2'),
    layers.Dense(16, activation='relu',                    name='Hidden_3'),
    layers.Dense(3,  activation='softmax',                 name='Output'),
], name='Iris_NN')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=0
)

actual_epochs = len(history.history['loss'])
best_epoch    = np.argmin(history.history['val_loss'])

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(f"\nTraining stopped at epoch {actual_epochs} (best: {best_epoch + 1})")
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"\n{classification_report(yi_test, y_pred, target_names=le.classes_)}")

epochs_range = range(1, actual_epochs + 1)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Neural Network - Iris", fontsize=14, fontweight='bold')

axes[0].plot(epochs_range, history.history['loss'],     'b-',  lw=2, label='Train Loss')
axes[0].plot(epochs_range, history.history['val_loss'], 'r--', lw=2, label='Val Loss')
axes[0].axvline(best_epoch + 1, color='green', linestyle=':', alpha=0.7)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history.history['accuracy'],     'b-',  lw=2, label='Train Acc')
axes[1].plot(epochs_range, history.history['val_accuracy'], 'r--', lw=2, label='Val Acc')
axes[1].axhline(test_acc, color='green', linestyle=':', alpha=0.7, label=f'Test={test_acc:.3f}')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training & Validation Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

cm = confusion_matrix(yi_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[2])
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")
axes[2].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("outputs/L3_T3_neural_network.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to outputs/")
