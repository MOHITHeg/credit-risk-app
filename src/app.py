import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. LOAD YOUR SAVED ARTIFACTS ---
# We use st.cache_resource to load these only once
@st.cache_resource
def load_artifacts():
    """
    Loads the saved preprocessor and model from the 'artifacts' folder.
    """
    try:
        preprocessor = joblib.load('artifacts/preprocessor.joblib')
        model = joblib.load('artifacts/best_xgb_final.joblib')
        return preprocessor, model
    except FileNotFoundError:
        st.error("ERROR: Model or Preprocessor file not found in 'artifacts/' folder.")
        st.stop()

preprocessor, model = load_artifacts()

# --- 2. RE-CREATE YOUR FEATURE ENGINEERING FUNCTION ---
# This MUST be the exact same function from your training script.
def engineer_features(df):
    """
    Apply feature engineering exactly as in the training pipeline.
    """
    df_fe = df.copy()
   
    # A) Create 'credit_per_month' (fix division by zero)
    df_fe['credit_per_month'] = df_fe['Credit amount'] / df_fe['Duration']
    df_fe['credit_per_month'] = df_fe['credit_per_month'].replace([np.inf, -np.inf], 0)
   
    # B) Create 'Age_bin'
    df_fe['Age_bin'] = pd.cut(
        df_fe['Age'],
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-100'],
        right=True
    )
   
    # C) Combine 'Purpose' categories
    relevant_purposes = ['education', 'domestic appliances']
    df_fe['Purpose_combined'] = df_fe['Purpose'].apply(
        lambda x: 'education_or_domestic' if x in relevant_purposes else 'other_purpose'
    )
   
    return df_fe

# --- 3. CREATE THE STREAMLIT UI ---

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("Credit Risk Prediction")
st.write("""
This app predicts whether a loan applicant is a **Good Risk** (will repay) or **Bad Risk** (will default)
using your trained XGBoost model. Please fill in the applicant's details on the left.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Applicant Information")

# Create a dictionary to hold all the inputs
# We need to use the RAW column names, *before* feature engineering
input_data = {}

# --- Input Fields ---
# (I've used the common categories from the dataset)
input_data['Checking account'] = st.sidebar.selectbox(
    'Checking Account Status',
    ['little', 'moderate', 'no checking account', 'rich']
)

input_data['Duration'] = st.sidebar.slider('Loan Duration (in months)', 4, 72, 24)
input_data['Credit amount'] = st.sidebar.number_input('Credit Amount (€)', 250, 20000, 2500)
input_data['Age'] = st.sidebar.slider('Age', 19, 75, 35)

input_data['Credit history'] = st.sidebar.selectbox(
    'Credit History',
    ['no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit']
)

input_data['Purpose'] = st.sidebar.selectbox(
    'Purpose',
    ['radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'domestic appliances', 'repairs', 'vacation/others']
)

input_data['Saving accounts'] = st.sidebar.selectbox(
    'Savings Account',
    ['little', 'moderate', 'rich', 'quite rich', 'no savings account']
)

input_data['Present employment since'] = st.sidebar.selectbox(
    'Employment Since',
    ['unemployed', '<1', '1-4', '4-7', '>=7']
)

input_data['Sex'] = st.sidebar.radio('Sex', ['male', 'female'])
input_data['Housing'] = st.sidebar.selectbox('Housing', ['rent', 'own', 'for free'])
input_data['Job'] = st.sidebar.selectbox('Job', ['unskilled and non-resident', 'unskilled and resident', 'skilled', 'highly skilled'])

# --- 4. PREDICTION LOGIC ---

# Create a "Predict" button
if st.sidebar.button("Predict Credit Risk"):
   
    # 1. Convert input data to a DataFrame
    # We use a single row [0]
    input_df = pd.DataFrame([input_data])
    st.write("---")
    st.subheader("Processing Applicant Data...")

    try:
        # 2. Apply Feature Engineering
        st.write("1. Applying feature engineering...")
        fe_df = engineer_features(input_df)
       
        # 3. Apply Preprocessing
        st.write("2. Scaling and encoding data...")
        processed_df = preprocessor.transform(fe_df)
       
        # 4. Make Prediction
        st.write("3. Making prediction...")
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)
       
        # 5. Display Result
        st.write("---")
        st.subheader("Prediction Result")
       
        # Remember: 0 = 'good', 1 = 'bad'
        if prediction[0] == 0:
            st.success("✅ Good Risk")
            st.markdown(f"**Confidence:** {probability[0][0]*100:.2f}%")
            st.balloons()
        else:
            st.error("❌ Bad Risk")
            st.markdown(f"**Confidence:** {probability[0][1]*100:.2f}%")
            st.snow()

        with st.expander("Show Processed Data (for debugging)"):
            st.dataframe(processed_df)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
