import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. MEMUAT DATASET
# ==========================================
# Pastikan file CSV berada di direktori yang sama
df = pd.read_csv('Machine_Learning\data_cuaca_clean.csv')

# ==========================================
# 2. FEATURE ENGINEERING (MEMBUAT TARGET LABEL)
# ==========================================

# Fungsi Klasifikasi Skala Beaufort
def get_beaufort_class(speed_ms):

    if speed_ms < 0.5: return '0 - Tenang'
    elif speed_ms <= 1.5: return '1 - Angin Sangat Lemah'
    elif speed_ms <= 3.3: return '2 - Angin Lemah'
    elif speed_ms <= 5.4: return '3 - Angin Sepoi-sepoi'
    elif speed_ms <= 7.9: return '4 - Angin Sedang'
    elif speed_ms <= 10.7: return '5 - Angin Cukup Kencang'
    elif speed_ms <= 13.8: return '6 - Angin Kencang'
    elif speed_ms <= 17.1: return '7 - Angin Sangat Kencang'
    elif speed_ms <= 20.7: return '8 - Angin Ribut'
    elif speed_ms <= 24.4: return '9 - Angin Ribut Kuat'
    elif speed_ms <= 28.4: return '10 - Badai'
    elif speed_ms <= 32.6: return '11 - Badai Hebat'
    else: return '12 - Badai Topan'

# Fungsi Klasifikasi Steadman Heat Index (Pendekatan formula NWS/NOAA)
def get_heat_index_class(t_celsius, rh):
    # Konversi Celcius ke Fahrenheit
    t_f = (t_celsius * 1.8) + 32
    
    # Formula dasar Steadman Heat Index
    hi_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    
    # Jika Heat Index > 80F, gunakan regresi Rothfusz untuk akurasi penuh
    if hi_f >= 80:
        hi_f = -42.379 + 2.04901523*t_f + 10.14333127*rh \
               - 0.22475541*t_f*rh - 0.00683783*t_f*t_f - 0.05481717*rh*rh \
               + 0.00122874*t_f*t_f*rh + 0.00085282*t_f*rh*rh - 0.00000199*t_f*t_f*rh*rh
               
    # Konversi kembali ke Celcius untuk klasifikasi kategori
    hi_c = (hi_f - 32) / 1.8
    
    # Kategori peringatan bahaya panas (Heat Index)
    if hi_c < 27: return 'Normal'
    elif hi_c <= 32: return 'Panas'
    elif hi_c <= 41: return 'Sangat Panas'
    elif hi_c <= 54: return 'Panas Ekstrem'
    else: return 'Panas Sangat Ekstrem'

# Menerapkan fungsi ke dataset untuk membuat kolom tabel baru
df['Beaufort_Class'] = df['Wind_Speed'].apply(get_beaufort_class)
df['Heat_Index_Class'] = df.apply(lambda row: get_heat_index_class(row['Dry_Bulb_Temperature'], row['Relative_Humidity']), axis=1)

# ==========================================
# 3. PERSIAPAN DATA & TRAINING MODEL
# ==========================================

# Menentukan Fitur (X) dan Target (y)
X = df[['Dry_Bulb_Temperature', 'Relative_Humidity', 'Wind_Speed']]
y_beaufort = df['Beaufort_Class']
y_heat = df['Heat_Index_Class']

# Membagi data menjadi Data Latih (80%) dan Data Uji (20%)
X_train, X_test, y_train_b, y_test_b = train_test_split(
    X, y_beaufort,
    test_size=0.2,
    random_state=42
)

_, _, y_train_h, y_test_h = train_test_split(
    X, y_heat,
    test_size=0.2,
    random_state=42
)

rf_config = {
    'n_estimators': 100,
    'criterion': 'gini',
    'max_depth': 17,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': 'sqrt', 
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': None,
    'random_state': 0,
    'verbose': 0,
    'warm_start': False,
    'class_weight': None,
    'ccp_alpha': 0.0,
    'max_samples': None
}

# Inisialisasi Model Random Forest (2 Model untuk 2 Output klasifikasi)
rf_beaufort = RandomForestClassifier(**rf_config)
rf_heat = RandomForestClassifier(**rf_config)

# Melatih Model
rf_beaufort.fit(X_train, y_train_b)
rf_heat.fit(X_train, y_train_h)

# Prediksi menggunakan data uji
y_pred_b = rf_beaufort.predict(X_test)
y_pred_h = rf_heat.predict(X_test)

cols = [
    'Dry_Bulb_Temperature',
    'Relative_Humidity',
    'Wind_Speed',
    'Beaufort_Class',
    'Heat_Index_Class'
]

print(df[cols])

# ==========================================
# 4. EVALUASI MODEL (Accuracy, Precision, Recall, F1-Score)
# ==========================================

print("=== EVALUASI MODEL: SKALA BEAUFORT ===")
print(classification_report(y_test_b, y_pred_b, zero_division=0))

print("\n=== EVALUASI MODEL: HEAT INDEX ===")
print(classification_report(y_test_h, y_pred_h, zero_division=0))

model_terpadu = {
    'model_beaufort': rf_beaufort,
    'model_heat_index': rf_heat
}
joblib.dump(model_terpadu, 'model_cuaca_80-20.pkl')
print("\n[INFO] Model Random Forest berhasil disimpan dalam format .pkl!")

# ==========================================
# 5. VISUALISASI CONFUSION MATRIX SEABORN
# ==========================================

# Menghitung Confusion Matrix
cm_beaufort = confusion_matrix(y_test_b, y_pred_b, labels=rf_beaufort.classes_)
cm_heat = confusion_matrix(y_test_h, y_pred_h, labels=rf_heat.classes_)

# # Membuat plot satu figur dengan 2 kolom (berdampingan)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Visualisasi Confusion Matrix Beaufort
sns.heatmap(cm_beaufort, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=rf_beaufort.classes_, yticklabels=rf_beaufort.classes_,
            cbar_kws={'shrink': 0.8})
axes[0].set_title('Confusion Matrix: Prediksi Skala Beaufort', fontsize=14, pad=15)
axes[0].set_xlabel('Nilai Prediksi', fontsize=12)
axes[0].set_ylabel('Nilai Aktual', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Visualisasi Confusion Matrix Heat Index
sns.heatmap(cm_heat, annot=True, fmt='d', cmap='Oranges', ax=axes[1], 
            xticklabels=rf_heat.classes_, yticklabels=rf_heat.classes_,
            cbar_kws={'shrink': 0.8})
axes[1].set_title('Confusion Matrix: Prediksi Heat Index', fontsize=14, pad=15)
axes[1].set_xlabel('Nilai Prediksi', fontsize=12)
axes[1].set_ylabel('Nilai Aktual', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
