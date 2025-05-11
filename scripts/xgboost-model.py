import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb

# Veri dosyasını yükle
data_path = "../data/cleaned_data.csv"
df = pd.read_csv(data_path)

# Özellik ve hedef değişken
X = df.drop('Fraud_Label', axis=1)
y = df['Fraud_Label']

# Eğitim/test ayırımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modeli tanımla ve eğit
model = XGBClassifier(
    max_depth=3,        # daha sığ ağaçlar
    n_estimators=100,   # daha az ağaç
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Performans metrikleri
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === CROSS VALIDATION (F1) ===

cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print("Cross-validation F1 scores:", cv_scores)
print("Mean CV F1 score:", cv_scores.mean())

xgb.plot_importance(model, max_num_features=10)
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance grafiği kaydedildi.")

y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)[:, 1]
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

print("TRAIN SET")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("F1 Score:", f1_score(y_train, y_train_pred))
print("ROC-AUC:", roc_auc_score(y_train, y_train_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("\nClassification Report:\n", classification_report(y_train, y_train_pred))


