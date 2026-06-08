# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import joblib

print("Mulai memproses data...")

# 1. Load Dataset
df = pd.read_csv('test.csv', sep=';')

# 2. Pembersihan Awal
if 'Column1' in df.columns:
    df.drop(['Column1'], axis=1, inplace=True)
if 'id' in df.columns:
    df.drop(['id'], axis=1, inplace=True)

# 3. Label Encoding untuk Fitur Kategorikal
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Target label encoding secara terpisah
le_target = LabelEncoder()
df['satisfaction'] = le_target.fit_transform(df['satisfaction'])
encoders['satisfaction'] = le_target

# 4. Memisahkan Fitur dan Target
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# 5. Imputasi & Normalisasi Data
numeric_cols = X.columns.tolist()

imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=numeric_cols)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=numeric_cols)

# 6. Mengatasi Imbalance Data dengan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 7. Pelatihan Model Akhir dengan Random Forest
# Parameter di bawah menggunakan hasil tuning terbaik dari GridSearchCV
model_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model_rf.fit(X_resampled, y_resampled)

print("Model berhasil dilatih dengan sukses!")

# 8. Menyimpan Objek Prapemrosesan dan Model ke File pkl
artifacts = {
    'model': model_rf,
    'encoders': encoders,
    'imputer': imputer,
    'scaler': scaler,
    'feature_names': X.columns.tolist()
}

# Ubah compress=3 (ini akan menekan ukuran file secara signifikan)
joblib.dump(artifacts, 'model_artifacts.pkl', compress=3)
print("File 'model_artifacts.pkl' berhasil disimpan. Siap digunakan di Streamlit!")