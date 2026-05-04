# ==========================================================
# DETECÇÃO DE ATAQUES IIoT (PARQUET + PIPELINE CORRETO)
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ==========================================================
# 1. CAMINHOS (GARANTIDO)
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "datasense_full_train_1sec.parquet")
test_path  = os.path.join(BASE_DIR, "datasense_full_test_1sec.parquet")

print("Train:", train_path)
print("Test :", test_path)

# ==========================================================
# 2. CARREGAMENTO
# ==========================================================

df_train = pd.read_parquet(train_path)
df_test  = pd.read_parquet(test_path)

print("\n===== DATASET =====")
print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)

# ==========================================================
# 3. TARGET
# ==========================================================

def get_target(df):
    if "label" in df.columns:
        return "label"
    elif "attack" in df.columns:
        return "attack"
    elif "class" in df.columns:
        return "class"
    else:
        raise Exception("Coluna alvo não encontrada!")

target_col = get_target(df_train)

# ==========================================================
# 4. LIMPEZA
# ==========================================================

df_train = df_train.dropna()
df_test  = df_test.dropna()

# ==========================================================
# 5. SEPARAÇÃO
# ==========================================================

X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]

X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col]

# ==========================================================
# 6. ONE-HOT ENCODING (SEGURO)
# ==========================================================

X_train = pd.get_dummies(X_train)
X_test  = pd.get_dummies(X_test)

# alinhar colunas (CRÍTICO)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nDistribuição das classes:")
print(y_train.value_counts())

# ==========================================================
# 7. GRÁFICO DE CLASSES
# ==========================================================

plt.figure()
y_train.value_counts().plot(kind='bar')
plt.title("Distribuição das Classes (Treino)")
plt.tight_layout()
plt.savefig("grafico_classes.png")
plt.close()

# ==========================================================
# 8. PIPELINE (SEM VAZAMENTO)
# ==========================================================

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        max_iter=2000,
        class_weight='balanced'
    ))
])

# ==========================================================
# 9. TREINAMENTO
# ==========================================================

pipeline.fit(X_train, y_train)

# ==========================================================
# 10. CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='f1'
)

print("\nF1-score (cross-validation):")
print("Scores:", scores)
print("Média:", scores.mean())

# ==========================================================
# 11. PREDIÇÃO
# ==========================================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# ==========================================================
# 12. MÉTRICAS
# ==========================================================

print("\n===== RESULTADOS =====")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ==========================================================
# 13. MATRIZ DE CONFUSÃO
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusao.png")
plt.close()

# ==========================================================
# 14. CURVA ROC
# ==========================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.tight_layout()
plt.savefig("curva_roc.png")
plt.close()

# ==========================================================
# 15. IMPORTÂNCIA DAS FEATURES
# ==========================================================

model = pipeline.named_steps['model']
importance = model.coef_[0]

feat_imp = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

print("\nTop 10 Features mais importantes:")
print(feat_imp.head(10))

# ==========================================================
# 16. GRÁFICO DE IMPORTÂNCIA
# ==========================================================

plt.figure()
sns.barplot(
    x="Importance",
    y="Feature",
    data=feat_imp.head(10)
)
plt.title("Top 10 Features mais importantes")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ==========================================================
# FINAL
# ==========================================================

print("\nArquivos gerados:")
print("- grafico_classes.png")
print("- matriz_confusao.png")
print("- curva_roc.png")
print("- feature_importance.png")