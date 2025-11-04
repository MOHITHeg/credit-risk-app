# German Credit Risk Prediction

This project is a complete machine learning pipeline to predict credit risk using the German Credit Risk dataset. The goal is to build a reliable and balanced classification model that can identify 'good' (low risk) and 'bad' (high risk) credit applicants.

The final model is a highly tuned **XGBoost Classifier** that achieves **79% accuracy** with a strong, balanced F1-score for both classes.

---

## 🎯 Final Model Performance

This champion model was achieved after a rigorous process of feature engineering, imbalance handling, and hyperparameter tuning. It demonstrates a strong balance between predicting the majority (bad risk) and minority (good risk) classes.

**Final Accuracy: 0.79**

**Final Confusion Matrix:**
[[118  23]
[ 19  40]]

** Final Classification Report **
support

       0       0.86      0.84      0.85       141
       1       0.63      0.68      0.66        59

accuracy                           0.79       200

*(Note: Class 0 = 'good' risk, Class 1 = 'bad' risk)*

## 🚀 Project Pipeline

This model was built using a systematic process to ensure high performance and reliability.

### 1. Data Preprocessing
A robust `ColumnTransformer` pipeline was built to handle all preprocessing steps, including:
* **Imputation:** Filling missing values for numerical and categorical features.
* **Scaling:** Using `StandardScaler` on all numerical columns.
* **Encoding:** Applying `OneHotEncoder` and `OrdinalEncoder` to the categorical and ordinal features.

### 2. Advanced Feature Engineering
To improve the model's predictive power, several new features were created:
* **`credit_per_month`**: A ratio of `Credit amount` / `Duration` to capture monthly payment pressure.
* **`Age_bin`**: Grouping applicants into age brackets (e.g., '18-25', '26-35') to find non-linear patterns.
* **`Purpose_combined`**: Consolidating rare 'Purpose' categories to reduce noise.

### 3. Handling Class Imbalance (A Hybrid Approach)
The dataset is imbalanced (70% good risk, 30% bad risk). A powerful hybrid technique was used to create a highly balanced model:
* **SMOTE (Synthetic Minority Over-sampling Technique):** The training data was first resampled using SMOTE to create a 50/50 balance.
* **`scale_pos_weight`:** The XGBoost model was *also* given this parameter. This unconventional combination (balancing the data *and* applying a penalty) proved to be the key strategy, forcing the model to pay extra attention to the minority class and leading to the best F1-scores.

### 4. Hyperparameter Tuning
Multiple models (including `RandomForest`) were tested. The champion `XGBoost` model's parameters were then aggressively tuned using **Optuna** to find the optimal combination for maximum performance.

---

## 🛠️ How to Run

1.  Clone this repository.
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the training pipeline (this will train the model and save the artifacts):
    ```bash
    python3 src/Model_training.py
    ```
