import pandas as pd
import os

# =========================
# CAMINHO
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "datasense_full_train_1sec.parquet")

# =========================
# CARREGAR
# =========================
df = pd.read_parquet(train_path)

print("\n===== SHAPE =====")
print(df.shape)

# =========================
# COLUNAS
# =========================
print("\n===== COLUNAS =====")
for col in df.columns:
    print(col)

# =========================
# TIPOS DE DADOS
# =========================
print("\n===== TIPOS =====")
print(df.dtypes)

# =========================
# POSSÍVEIS TARGETS
# =========================
print("\n===== POSSÍVEIS TARGETS =====")

for col in df.columns:
    valores_unicos = df[col].nunique()
    
    # heurística: coluna com poucos valores únicos pode ser target
    if valores_unicos <= 10:
        print(f"\nColuna: {col}")
        print(df[col].value_counts().head(10))