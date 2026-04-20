# # # app.py
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from modules import (
# #     process_categorical,
# #     clean_missing_values,
# #     preserve_none,
# #     handle_outliers,
# #     get_custom_stats
# # )

# # st.set_page_config(page_title="SmartCleaner", layout="wide")
# # st.title("🧹 Smart Data Cleaner")

# # uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type="csv")

# # if uploaded_file:
# #     df = pd.read_csv(uploaded_file)
# #     df = preserve_none(df)

# #     st.subheader("🔍 Raw Dataset")
# #     st.dataframe(df, use_container_width=True)

# #     # Raw dataset stats
# #     st.subheader("📈 Raw Dataset Statistical Summary")
# #     raw_stats = get_custom_stats(df)
# #     st.dataframe(raw_stats, use_container_width=True)

# #     st.markdown("---")

# #     # Step 1: Categorical Encoding
# #     st.subheader("1️⃣ Categorical Encoding Setup")
# #     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# #     cat_cols_lower = [col.lower() for col in cat_cols]
# #     cat_cols_display = cat_cols_lower + ["NA"]

# #     selected_col_display = st.selectbox("Select categorical column for custom order or NA to auto-handle:",
# #                                         options=cat_cols_display)

# #     if selected_col_display != "NA":
# #         selected_col = cat_cols[cat_cols_lower.index(selected_col_display)]
# #     else:
# #         selected_col = "NA"

# #     predefined_order = ""
# #     proceed = False

# #     if selected_col == "NA":
# #         proceed = True
# #     elif selected_col != "NA":
# #         predefined_order = st.text_input("Enter predefined order (comma separated, e.g., low, medium, high):")
# #         if predefined_order.strip():
# #             proceed = True

# #     if proceed:
# #         df = process_categorical(df, selected_col, predefined_order)

# #         # Step 2: Outlier Detection
# #         st.subheader("2️⃣ Outlier Detection & Custom Handling")
# #         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# #         outlier_info = []

# #         for col in numeric_cols:
# #             Q1 = df[col].quantile(0.25)
# #             Q3 = df[col].quantile(0.75)
# #             IQR = Q3 - Q1
# #             lower = Q1 - 1.5 * IQR
# #             upper = Q3 + 1.5 * IQR
# #             outliers = ((df[col] < lower) | (df[col] > upper)).sum()
# #             if outliers > 0:
# #                 outlier_info.append({"Column": col, "Outliers Detected": int(outliers)})

# #         if outlier_info:
# #             st.markdown("### 📊 Columns with Outliers Detected")
# #             outlier_df = pd.DataFrame(outlier_info)
# #             st.dataframe(outlier_df, use_container_width=True)

# #             st.markdown("### 🛠️ Choose Outlier Handling Method per Column")
# #             method_selections = {}
# #             for row in outlier_info:
# #                 col = row["Column"]
# #                 method_selections[col] = st.selectbox(
# #                     f"Choose method for `{col}`:",
# #                     options=["keep", "remove", "cap"],
# #                     index=0,
# #                     key=col + "_method"
# #                 )

# #             if st.button("🚀 Clean Data Now"):
# #                 for col, method in method_selections.items():
# #                     if method != "keep":
# #                         df[[col]], _ = handle_outliers(df[[col]], method=method)

# #                 # Proceed to Step 3 below 👇
# #                 go_to_step3 = True
# #         else:
# #             st.success("✅ No outliers detected in numeric columns.")
# #             go_to_step3 = st.button("➡️ Proceed to Step 3 (Missing Value Handling)")

# #         # Step 3: Missing Value Handling
# #         if 'go_to_step3' in locals() and go_to_step3:
# #             st.subheader("3️⃣ Missing Value Handling based on Kurtosis")
# #             df, kurtosis_logs, summary_df = clean_missing_values(df)
# #             for msg in kurtosis_logs:
# #                 st.markdown(msg)

# #             if not summary_df.empty:
# #                 st.markdown("### 📊 Missing Value Imputation Summary")
# #                 st.dataframe(summary_df, use_container_width=True)

# #             # Convert bool to int
# #             bool_cols = df.select_dtypes(include=["bool"]).columns
# #             df[bool_cols] = df[bool_cols].astype(int)

# #             # Final Cleaned Data View
# #             st.markdown("---")
# #             st.subheader("✅ Cleaned Dataset (Numeric View)")
# #             st.dataframe(df, use_container_width=True)

# #             # Cleaned dataset stats
# #             st.subheader("📉 Cleaned Dataset Statistical Summary")
# #             cleaned_stats = get_custom_stats(df)
# #             st.dataframe(cleaned_stats, use_container_width=True)

# #             # Comparison Table
# #             st.subheader("🔍 Statistical Comparison: Raw - Cleaned = Difference")

# #             comparison_stats = pd.DataFrame()
# #             for col in raw_stats.columns:
# #                 comparison_stats[f"{col} (Raw - Cleaned) in percentage"] = (
# #                     abs(raw_stats[col] - cleaned_stats[col]) / raw_stats[col] * 100
# #                 )

# #             comparison_stats = comparison_stats.round(4)
# #             st.dataframe(comparison_stats, use_container_width=True)

# #             # Download Option
# #             @st.cache_data
# #             def convert_df(df):
# #                 return df.to_csv(index=False).encode('utf-8')

# #             csv = convert_df(df)

# #             st.download_button(
# #                 label="📥 Download Cleaned CSV",
# #                 data=csv,
# #                 file_name='cleaned_dataset.csv',
# #                 mime='text/csv',
# #             )
# #     else:
# #         st.warning("👆 Please complete the categorical setup above to continue.")



# import streamlit as st
# import pandas as pd
# import numpy as np
# from module import (
#     process_categorical,
#     preserve_none,
#     handle_outliers,
#     get_custom_stats,
#     smart_missing_handler,
#     apply_scaling,
#     generate_summary_report
# )

# st.set_page_config(page_title="SmartCleaner", layout="wide")
# st.title("🧹 Smart Data Cleaner")

# uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     df = preserve_none(df)

#     st.subheader("🔍 Raw Dataset")
#     st.dataframe(df, use_container_width=True)

#     raw_shape = df.shape
#     raw_stats = get_custom_stats(df)

#     st.subheader("📈 Raw Dataset Statistical Summary")
#     st.dataframe(raw_stats, use_container_width=True)
#     st.markdown("---")

#     # Step 1: Categorical Encoding
#     st.subheader("1️⃣ Categorical Encoding Setup")
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     if cat_cols:
#         cat_cols_lower = [col.lower() for col in cat_cols]
#         cat_cols_display = cat_cols_lower + ["NA"]
#         selected_col_display = st.selectbox("Select categorical column for custom order or NA:", options=cat_cols_display)
#         if selected_col_display != "NA":
#             selected_col = cat_cols[cat_cols_lower.index(selected_col_display)]
#             predefined_order = st.text_input("Enter predefined order (comma separated):")
#             if predefined_order.strip():
#                 df = process_categorical(df, selected_col, predefined_order)
#         else:
#             df = process_categorical(df, "NA", "")
#         encoding_summary = f"Categorical columns encoded: {cat_cols}"
#     else:
#         st.info("No categorical columns detected.")
#         encoding_summary = "No categorical columns"

#     # Step 2: Outlier Detection Column-wise
#     st.subheader("2️⃣ Outlier Detection & Handling")
#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     outlier_choices = {}
#     for col in numeric_cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         outliers = ((df[col] < lower) | (df[col] > upper)).sum()
#         if outliers > 0:
#             st.markdown(f"**Column:** `{col}` — Outliers detected: {outliers}")
#             choice = st.selectbox(f"Choose outlier handling for `{col}`:", ["keep", "remove", "cap"], key=f"out_{col}")
#             outlier_choices[col] = choice

#     if st.button("🚀 Apply Outlier Handling"):
#         for col, method in outlier_choices.items():
#             df, _ = handle_outliers(df[[col]].copy().join(df.drop(columns=[col])), method=method)
#         st.success("Outlier handling applied column-wise.")

#     # Step 3: Missing Value Handling
#     st.subheader("3️⃣ Missing Value Handling")
#     missing_info = df.isnull().mean() * 100
#     missing_cols = missing_info[missing_info > 0].round(2)
#     imputation_summary = {}

#     if not missing_cols.empty:
#         st.dataframe(missing_cols, use_container_width=True)
#         for col in missing_cols.index:
#             missing_pct = missing_info[col]

#             if missing_pct < 20:
#                 df, summary = smart_missing_handler(df, col, missing_pct)
#                 imputation_summary[col] = summary
#                 st.success(summary)

#             else:  # >20%
#                 st.warning(f"⚠️ Column `{col}` has >20% missing values.")
#                 important = st.radio(
#                     f"Is column `{col}` important?",
#                     ["Yes", "No"], key=f"imp_{col}"
#                 )
#                 if important == "No":
#                     df, summary = smart_missing_handler(df, col, missing_pct, {"important": False})
#                     imputation_summary[col] = summary
#                     st.info(summary)
#                 else:
#                     method = st.selectbox(
#                         f"Choose imputation method for `{col}`:",
#                         ["median", "knn", "mice", "regression"],
#                         key=f"method_{col}"
#                     )
#                     if st.button(f"Impute `{col}`", key=f"btn_{col}"):
#                         df, summary = smart_missing_handler(
#                             df, col, missing_pct,
#                             {"important": True, "method": method}
#                         )
#                         imputation_summary[col] = summary
#                         st.success(summary)
#     else:
#         imputation_summary = "No missing values"
#         st.success("No missing values found.")

#     # Step 4: Normalization / Scaling
#     st.subheader("4️⃣ Normalization / Scaling")
#     scaling_method = st.selectbox("Choose scaling method:", ["none", "minmax", "standard", "robust"])
#     if scaling_method != "none":
#         df = apply_scaling(df, method=scaling_method)
#         st.success(f"✅ Scaling applied using {scaling_method}")
#     else:
#         st.info("No scaling applied.")

#     # Step 5: Final Dataset Preview & Stats Comparison
#     st.subheader("5️⃣ Final Clean Dataset Preview")
#     st.dataframe(df, use_container_width=True)

#     cleaned_stats = get_custom_stats(df)

#     st.subheader("📊 Cleaned Dataset Statistical Summary")
#     st.dataframe(cleaned_stats, use_container_width=True)

#     # Statistical Comparison
#     comparison = ((raw_stats - cleaned_stats) / raw_stats) * 100
#     comparison = comparison.round(2).fillna(0)
#     st.subheader("📊 Statistical Comparison (% Difference)")
#     st.dataframe(comparison, use_container_width=True)

#     # Step 6: Download Cleaned Dataset + Report
#     st.subheader("6️⃣ Download Results")
#     @st.cache_data
#     def convert_df(df):
#         return df.to_csv(index=False).encode('utf-8')
#     csv = convert_df(df)

#     st.download_button(
#         label="📥 Download Cleaned CSV",
#         data=csv,
#         file_name='cleaned_dataset.csv',
#         mime='text/csv',
#     )

#     report = generate_summary_report(
#         df, raw_shape, encoding_summary, outlier_choices, scaling_method, imputation_summary
#     )
#     st.download_button(
#         label="📥 Download Cleaning Report",
#         data=str(report),
#         file_name='cleaning_report.txt',
#         mime='text/plain',
#     )

import streamlit as st
import pandas as pd
import numpy as np
from module import (
    process_categorical,
    preserve_none,
    handle_outliers,
    get_custom_stats,
    smart_missing_handler,
    apply_scaling,
    generate_summary_report
)

st.set_page_config(page_title="SmartCleaner", layout="wide")
st.title("🧹 Smart Data Cleaner")

uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = preserve_none(df)

    st.subheader("🔍 Raw Dataset")
    st.dataframe(df, use_container_width=True)

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
        else:
            df = process_categorical(df, "NA", "")
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

    # Statistical Comparison
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
