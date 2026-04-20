# chatbot_app.py

import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import numpy as np

# import your module functions
from module import (
    process_categorical,
    preserve_none,
    handle_outliers,
    get_custom_stats,
    smart_missing_handler,
    apply_scaling,
    generate_summary_report
)

# Configure Gemini
genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel('YOUR_MODEL_NAME_HERE')

st.set_page_config(page_title="Smart EDA Chatbot", layout="wide")
st.title("🤖 Smart EDA Chatbot")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type="csv")

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Raw Dataset Preview")
    st.dataframe(raw_df.head())

    # Step 2: User prompt for LLM
    user_problem = st.text_area("📝 Describe your problem (e.g., 'predict survival in Titanic dataset')")

    if st.button("✨ Run LLM to select important columns"):
        prompt = f"""
        You are an expert data scientist.

        The user’s problem is: "{user_problem}"

        The dataset columns are: {list(raw_df.columns)}

        Your task:

        1. Based on the problem, identify the most appropriate target column (only one).

        2. Select the useful feature columns that can serve as predictors for solving the problem. 
           - Exclude identifiers (IDs, serial numbers, indexes).
           - Exclude free-text notes or irrelevant metadata.
           - Exclude columns that directly leak the target (e.g., duplicates or derived labels).

        3. Return the result only in valid JSON format with the following structure:

        {{
          "target": "target_column_name",
          "features": ["list", "of", "features"]
        }}
        """

        response = model.generate_content(prompt)
        raw_output = response.text.strip()

        try:
            if raw_output.startswith("```"):
                raw_output = raw_output.strip("`").replace("json", "").strip()

            result = json.loads(raw_output)
            target_col = result["target"]
            feature_cols = result["features"]

            st.success(f"🎯 Target: {target_col}")
            st.success(f"📊 Features: {feature_cols}")

            # Step 3: Create modified dataset
            cols_to_use = [target_col] + feature_cols
            df = raw_df[cols_to_use].copy()
            df = preserve_none(df)

            st.subheader("🧩 Modified Dataset (Target + Features)")
            st.dataframe(df.head())

            # ==========================
            # Now reuse app.py SmartCleaner pipeline
            # ==========================

            raw_shape = df.shape
            raw_stats = get_custom_stats(df)

            st.subheader("📈 Raw Dataset Statistical Summary")
            st.dataframe(raw_stats, use_container_width=True)
            st.markdown("---")

            # Step 1: Categorical Encoding
            st.subheader("1️⃣ Categorical Encoding Setup")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                cat_cols_lower = [col.lower() for col in cat_cols]
                cat_cols_display = cat_cols_lower + ["NA"]
                selected_col_display = st.selectbox("Select categorical column for custom order or NA:", options=cat_cols_display)
                if selected_col_display != "NA":
                    selected_col = cat_cols[cat_cols_lower.index(selected_col_display)]
                    predefined_order = st.text_input("Enter predefined order (comma separated):")
                    if predefined_order.strip():
                        df = process_categorical(df, selected_col, predefined_order)
    
                encoding_summary = f"Categorical columns encoded: {cat_cols}"
            else:
                st.info("No categorical columns detected.")
                encoding_summary = "No categorical columns"

            # Step 2: Outlier Detection Column-wise
            st.subheader("2️⃣ Outlier Detection & Handling")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            outlier_choices = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if outliers > 0:
                    st.markdown(f"**Column:** `{col}` — Outliers detected: {outliers}")
                    choice = st.selectbox(f"Choose outlier handling for `{col}`:", ["keep", "remove", "cap"], key=f"out_{col}")
                    outlier_choices[col] = choice

            if st.button("🚀 Apply Outlier Handling"):
                for col, method in outlier_choices.items():
                    df, _ = handle_outliers(df[[col]].copy().join(df.drop(columns=[col])), method=method)
                st.success("Outlier handling applied column-wise.")

            # Step 3: Missing Value Handling
            st.subheader("3️⃣ Missing Value Handling")
            missing_info = df.isnull().mean() * 100
            missing_cols = missing_info[missing_info > 0].round(2)
            imputation_summary = {}

            if not missing_cols.empty:
                st.dataframe(missing_cols, use_container_width=True)
                for col in missing_cols.index:
                    missing_pct = missing_info[col]

                    if missing_pct < 20:
                        df, summary = smart_missing_handler(df, col, missing_pct)
                        imputation_summary[col] = summary
                        st.success(summary)
                    else:  # >20%
                        st.warning(f"⚠️ Column `{col}` has >20% missing values.")
                        important = st.radio(
                            f"Is column `{col}` important?",
                            ["Yes", "No"], key=f"imp_{col}"
                        )
                        if important == "No":
                            df, summary = smart_missing_handler(df, col, missing_pct, {"important": False})
                            imputation_summary[col] = summary
                            st.info(summary)
                        else:
                            method = st.selectbox(
                                f"Choose imputation method for `{col}`:",
                                ["median", "knn", "mice", "regression"],
                                key=f"method_{col}"
                            )
                            if st.button(f"Impute `{col}`", key=f"btn_{col}"):
                                df, summary = smart_missing_handler(
                                    df, col, missing_pct,
                                    {"important": True, "method": method}
                                )
                                imputation_summary[col] = summary
                                st.success(summary)
            else:
                imputation_summary = "No missing values"
                st.success("No missing values found.")

            # Step 4: Normalization / Scaling
            st.subheader("4️⃣ Normalization / Scaling")
            scaling_method = st.selectbox("Choose scaling method:", ["none", "minmax", "standard", "robust"])
            if scaling_method != "none":
                df = apply_scaling(df, method=scaling_method)
                st.success(f"✅ Scaling applied using {scaling_method}")
            else:
                st.info("No scaling applied.")

            # 🔹 Final NaN Cleanup
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna("MISSING", inplace=True)

            # Step 5: Final Dataset Preview & Stats Comparison
            st.subheader("5️⃣ Final Clean Dataset Preview")
            st.dataframe(df, use_container_width=True)

            cleaned_stats = get_custom_stats(df)

            st.subheader("📊 Cleaned Dataset Statistical Summary")
            st.dataframe(cleaned_stats, use_container_width=True)

            comparison = ((raw_stats - cleaned_stats) / raw_stats) * 100
            comparison = comparison.round(2).fillna(0)
            st.subheader("📊 Statistical Comparison (% Difference)")
            st.dataframe(comparison, use_container_width=True)

            # Step 6: Download Cleaned Dataset + Report
            st.subheader("6️⃣ Download Results")
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            csv = convert_df(df)

            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name='cleaned_dataset.csv',
                mime='text/csv',
            )

            report = generate_summary_report(
                df, raw_shape, encoding_summary, outlier_choices, scaling_method, imputation_summary
            )
            st.download_button(
                label="📥 Download Cleaning Report",
                data=str(report),
                file_name='cleaning_report.txt',
                mime='text/plain',
            )

        except Exception as e:
            st.error(f"⚠️ Could not parse Gemini response. Raw output:\n\n{raw_output}\n\nError: {e}")
