import streamlit as st
import pandas as pd
from modules import CLASSIFICATION_MODELS, REGRESSION_MODELS, train_and_evaluate

st.set_page_config(page_title="AutoML App", layout="wide")
st.title("🤖 AutoML Streamlit App")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("📂 Upload your cleaned dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset Loaded Successfully!")
    st.dataframe(df.head())

    # Step 2: Select Target Variable
    target = st.selectbox("🎯 Select Target Variable", df.columns)

    # Auto-detect features
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]

    # Step 3: Select Problem Type
    problem_type = st.radio("⚡ Select Problem Type", ["Classification", "Regression"])

    # Step 4: Select Models (Dynamic based on problem type)
    if problem_type == "Classification":
        model_options = list(CLASSIFICATION_MODELS.keys())
    else:
        model_options = list(REGRESSION_MODELS.keys())

    selected_models = st.multiselect("🧠 Select Models to Train", model_options)

    # Step 5: Train and Evaluate
    if st.button("🚀 Run AutoML"):
        if selected_models:
            results = train_and_evaluate(X, y, problem_type, selected_models)
            st.subheader("📊 Model Results")
            for model, metrics in results.items():
                st.write(f"### {model}")
                st.json(metrics)
        else:
            st.warning("⚠️ Please select at least one model to train.")
