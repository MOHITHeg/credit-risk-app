import os
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# === Load config ===
config_path = os.path.join(os.getcwd(), "config.yaml")
print(f"Loading configuration from: {config_path}")


with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

if config is None:
    raise ValueError("Configuration file is empty or not properly formatted.")
print("Configuration loaded successfully.")

# === Paths ===
data_path = config['artifacts']['data_ingestion']['raw_data_path']
preprocessor_path = config['artifacts']['model_training']['preprocessor_path']
trained_model_path = config['artifacts']['model_training']['trained_model_path']

# === Load Data ===
df = pd.read_csv(data_path)
# --- Feature engineering (must match what the preprocessor was fitted on) ---
# A) credit_per_month
df['credit_per_month'] = df['Credit amount'] / df['Duration']
df['credit_per_month'] = df['credit_per_month'].replace([np.inf, -np.inf], np.nan).fillna(0)

# B) Age_bin (use same bins/labels as in notebook)
df['Age_bin'] = pd.cut(df['Age'],
                       bins=[18, 25, 35, 45, 55, 65, 100],
                       labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-100'],
                       right=True)

# C) Purpose_combined (same grouping used previously)
relevant_purposes = ['education', 'domestic appliance']
df['Purpose_combined'] = df['Purpose'].apply(
    lambda x: 'education_or_domestic' if x in relevant_purposes else 'other_purpose'
)

X = df.drop(columns= 'Risk', axis=1, errors='ignore')
#y_raw= df['Risk']
y = df['Risk'].map({'good':0, 'bad':1})

# === Load Preprocessor ===
preprocessor = joblib.load(preprocessor_path)

# === Transform Data ===
# 1) Split RAW X/y first (keep stratify to match notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) Transform raw train/test with the loaded preprocessor
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 3) Apply SMOTE only to the transformed training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_transformed, y_train)




# === Load Model ===

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
best_params = {
    'n_estimators': 310,
    'scale_pos_weight': scale_pos_weight,
    'max_depth': 6,
    'learning_rate': 0.089,
    'subsample': 0.97,
    'colsample_bytree': 0.82,
}

best_xgb_final = XGBClassifier(**best_params, eval_metric='logloss',random_state=42)
best_xgb_final.fit(X_train_res, y_train_res)

# save model
os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
joblib.dump(best_xgb_final, trained_model_path)
print(f"Trained model saved at: {trained_model_path}")

# === Evaluate Model ===
y_pred = best_xgb_final.predict(X_test_transformed)

print("\n---  Final model accuracy ---", accuracy_score(y_test, y_pred))
print("\nFinal Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nFinal Classification Report:\n", classification_report(y_test, y_pred))
