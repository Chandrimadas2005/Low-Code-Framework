# #modules.Py
# import pandas as pd 
# import numpy as np
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import OrdinalEncoder
# from scipy.stats import kurtosis

# def process_categorical(df, selected_col, predefined_order):
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     for col in cat_cols:
#         df[col] = df[col].astype(str).str.lower()

#     if selected_col != "NA" and predefined_order:
#         selected_col = selected_col.lower()
#         order = [x.strip().lower() for x in predefined_order.split(",")]
#         if selected_col in df.columns:
#             enc = OrdinalEncoder(categories=[order])
#             df[selected_col] = enc.fit_transform(df[[selected_col]])
#             cat_cols.remove(selected_col)

#     for col in cat_cols:
#         unique_vals = df[col].nunique()
#         if unique_vals < 10:
#             df = pd.get_dummies(df, columns=[col], prefix=col)
#         else:
#             df[col] = pd.factorize(df[col])[0]
#     return df

# def clean_missing_values(df):
#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     messages = []
#     knn_needed = False

#     kurtosis_dict = {}
#     method_dict = {}

#     for col in numeric_cols:
#         if df[col].isnull().sum() > 0:
#             try:
#                 k = kurtosis(df[col].dropna(), fisher=False)
#                 kurtosis_dict[col] = k
#             except Exception as e:
#                 messages.append(f"⚠️ Column `{col}` — Kurtosis error: {str(e)} ➜ Skipping")

#     for col, k in kurtosis_dict.items():
#         msg = f"🔍 Column `{col}` — Kurtosis = {k:.2f} ➜ "
#         if k > 3:
#             df[col].fillna(df[col].median(), inplace=True)
#             method_dict[col] = ("Median", k)
#             msg += "Using **Median Imputation** (Leptokurtic > 3)"
#         elif abs(k - 3) < 0.1:
#             df[col].fillna(df[col].median(), inplace=True)
#             method_dict[col] = ("Median", k)
#             msg += "Using **Median Imputation** (≈ Normal)"
#         else:
#             knn_needed = True
#             method_dict[col] = ("KNN", k)
#             msg += "Will apply **KNN Imputation** (Platykurtic < 3)"
#         messages.append(msg)

#     if knn_needed:
#         knn_cols = [col for col, (method, _) in method_dict.items() if method == "KNN"]
#         imputer = KNNImputer(n_neighbors=3)
#         df[knn_cols] = imputer.fit_transform(df[knn_cols])
#         messages.append("🚀 KNN Imputation applied to selected numeric columns.")

#     summary_data = [{"Column": col, "Kurtosis": round(k, 2), "Method": method} 
#                     for col, (method, k) in method_dict.items()]
#     summary_df = pd.DataFrame(summary_data)

#     return df, messages, summary_df

# def handle_outliers(df, method="remove"):
#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     outlier_summary = []

#     for col in numeric_cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         before = df[col].isna().sum()
#         outliers = ((df[col] < lower) | (df[col] > upper)).sum()

#         if method == "remove":
#             # Replace outlier values with NaN instead of dropping rows
#             df[col] = df[col].mask((df[col] < lower) | (df[col] > upper), np.nan)
#         elif method == "cap":
#             df[col] = np.where(df[col] < lower, lower, df[col])
#             df[col] = np.where(df[col] > upper, upper, df[col])
#         # "keep" = no changes

#         after = df[col].isna().sum()
#         outlier_summary.append({
#             "Column": col,
#             "Outliers Detected": int(outliers),
#             "Method": method,
#             "Nulls Before": before,
#             "Nulls After": after
#         })

#     return df, pd.DataFrame(outlier_summary)

# def preserve_none(df):
#     missing_values = ["None", "none", "null", "Null", "NULL", "na", "NA", "NaN", "nan", "", " "]
#     df.replace(missing_values, np.nan, inplace=True)
#     return df

# def get_custom_stats(df):
#     numeric_df = df.select_dtypes(include=np.number)
#     stats = pd.DataFrame()

#     stats["Count"] = numeric_df.count()
#     stats["Mean"] = numeric_df.mean()
#     stats["Median"] = numeric_df.median()
#     stats["Mode"] = numeric_df.mode().iloc[0]  # first mode row
#     stats["Std"] = numeric_df.std()
#     stats["Variance"] = numeric_df.var()
#     stats["Kurtosis"] = numeric_df.kurtosis()

#     return stats.round(3)

# def apply_scaling(df, method="standard"):
#     df = df.copy()
#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     scaler = None

#     if method == "standard":
#         from sklearn.preprocessing import StandardScaler
#         scaler = StandardScaler()
#     elif method == "minmax":
#         from sklearn.preprocessing import MinMaxScaler
#         scaler = MinMaxScaler()

#     if scaler:
#         df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#     return df




import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis


# ---------------- Categorical Handling ---------------- #

def process_categorical(df, selected_col, predefined_order):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower()

    if selected_col != "NA" and predefined_order:
        selected_col = selected_col.lower()
        order = [x.strip().lower() for x in predefined_order.split(",")]
        if selected_col in df.columns:
            enc = OrdinalEncoder(categories=[order])
            df[selected_col] = enc.fit_transform(df[[selected_col]])
            cat_cols.remove(selected_col)

    for col in cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals < 10:
            df = pd.get_dummies(df, columns=[col], prefix=col)
        else:
            df[col] = pd.factorize(df[col])[0]
    return df


def preserve_none(df):
    missing_values = ["None", "none", "null", "Null", "NULL", "na", "NA", "NaN", "nan", "", " "]
    df.replace(missing_values, np.nan, inplace=True)
    return df


# ---------------- Outlier Handling ---------------- #

def handle_outliers(df, method="remove"):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    outlier_summary = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = df[col].isna().sum()
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        if method == "remove":
            df[col] = df[col].mask((df[col] < lower) | (df[col] > upper), np.nan)
        elif method == "cap":
            df[col] = np.where(df[col] < lower, lower, df[col])
            df[col] = np.where(df[col] > upper, upper, df[col])
        # keep → no changes

        after = df[col].isna().sum()
        outlier_summary.append({
            "Column": col,
            "Outliers Detected": int(outliers),
            "Method": method,
            "Nulls Before": before,
            "Nulls After": after
        })

    return df, pd.DataFrame(outlier_summary)


# ---------------- Smart Missing Value Handling ---------------- #

def smart_missing_handler(df, col, missing_pct, user_choice=None):
    """
    Handles missing values for a given column based on % missing and kurtosis.

    Parameters:
        df (pd.DataFrame): input dataframe
        col (str): column name
        missing_pct (float): % missing in column
        user_choice (dict or None): required if missing > 20%.
            Example: {"important": True, "method": "median"}

    Returns:
        df (pd.DataFrame): updated dataframe
        summary (str): description of what was done
    """
    if missing_pct < 20:
        try:
            k = kurtosis(df[col].dropna(), fisher=False)
            if k > 3 or abs(k - 3) < 0.1:
                df[col].fillna(df[col].median(), inplace=True)
                return df, f"Imputed `{col}` using Median (K={k:.2f})"
            else:
                imputer = KNNImputer(n_neighbors=3)
                df[[col]] = imputer.fit_transform(df[[col]])
                return df, f"Imputed `{col}` using KNN (K={k:.2f})"
        except Exception as e:
            return df, f"⚠️ Error calculating kurtosis for `{col}`: {e}"
    else:
        if not user_choice:
            return df, f"⚠️ Column `{col}` requires user input (important or not)."

        if not user_choice.get("important", True):
            df.drop(columns=[col], inplace=True)
            return df, f"Dropped `{col}` (>20% missing, not important)"

        method = user_choice.get("method", "median")
        if method == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif method == "knn":
            imputer = KNNImputer(n_neighbors=3)
            df[[col]] = imputer.fit_transform(df[[col]])
        elif method == "mice":
            imputer = IterativeImputer(random_state=0)
            df[[col]] = imputer.fit_transform(df[[col]])
        elif method == "regression":
            not_null = df[df[col].notnull()]
            null = df[df[col].isnull()]
            if not null.empty and len(not_null.columns) > 1:
                X_train = not_null.drop(columns=[col]).select_dtypes(include=np.number)
                y_train = not_null[col]
                X_pred = null.drop(columns=[col]).select_dtypes(include=np.number)

            if not X_train.empty and not X_pred.empty:
                from sklearn.impute import SimpleImputer
                imp = SimpleImputer(strategy="median")   # or "mean"
                X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
                X_pred = pd.DataFrame(imp.transform(X_pred), columns=X_pred.columns)

                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_pred)
                df.loc[df[col].isnull(), col] = preds

        return df, f"Imputed `{col}` using {method} (manual choice)"


# ---------------- Scaling ---------------- #

def apply_scaling(df, method="standard"):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    scaler = None
    if method == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

    if scaler:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# ---------------- Stats & Report ---------------- #

def get_custom_stats(df):
    numeric_df = df.select_dtypes(include=np.number)
    stats = pd.DataFrame()
    stats["Count"] = numeric_df.count()
    stats["Mean"] = numeric_df.mean()
    stats["Median"] = numeric_df.median()
    stats["Mode"] = numeric_df.mode().iloc[0]
    stats["Std"] = numeric_df.std()
    stats["Variance"] = numeric_df.var()
    stats["Kurtosis"] = numeric_df.kurtosis()
    return stats.round(3)


def generate_summary_report(df, raw_shape, encoding_summary, outlier_info, scaling_method, imputation_summary):
    report = {
        "Total Rows": df.shape[0],
        "Total Columns": df.shape[1],
        "Rows Changed": raw_shape[0] - df.shape[0],
        "Columns Changed": raw_shape[1] - df.shape[1],
        "Encoding Summary": encoding_summary,
        "Outliers Handled": outlier_info,
        "Scaling Method Used": scaling_method,
        "Imputation Summary": imputation_summary
    }
    return report
