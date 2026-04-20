import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# -----------------------------
# Helpers for Step 1: Sampling + Stationarity
# -----------------------------
def ensure_time_series(df: pd.DataFrame, time_col: str, value_col: str) -> pd.Series:
    out = df[[time_col, value_col]].copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[time_col, value_col]).sort_values(time_col)
    out = out.groupby(time_col, as_index=True)[value_col].mean()
    out.index.name = None
    return out

def pick_rule_from_gaps(dt_index: pd.DatetimeIndex) -> str:
    if len(dt_index) < 2:
        return "D"
    gaps = pd.Series(dt_index).diff().dropna()
    median_gap_days = (gaps.median() / pd.Timedelta(days=1))
    if median_gap_days <= 2:
        return "D"
    elif median_gap_days <= 14:
        return "W"
    else:
        return "M"

def auto_resample(series: pd.Series) -> tuple[pd.Series, str]:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    inferred = pd.infer_freq(s.index)
    if inferred is not None:
        if inferred.startswith("D"):
            rule = "D"
        elif inferred.startswith("W"):
            rule = "W"
        elif inferred.startswith("M"):
            rule = "M"
        else:
            rule = pick_rule_from_gaps(s.index)
    else:
        rule = pick_rule_from_gaps(s.index)
    resampled = s.resample(rule).mean()
    resampled = resampled.interpolate(method="time").ffill().bfill()
    return resampled, rule

# -----------------------------
# Helpers for Step 2: Residual-based Outlier Detection
# -----------------------------
def detect_outliers_residuals(series, period=None):
    series_clean = series.dropna().astype(float)
    if len(series_clean) < 10:
        return [], None, None, None
    
    try:
        # Auto-select seasonal period
        if period is None:
            freq = pd.infer_freq(series_clean.index)
            if freq:
                if freq.startswith('D'):
                    period = 7
                elif freq.startswith('W'):
                    period = 52
                elif freq.startswith('M'):
                    period = 12
                else:
                    period = min(len(series_clean) // 2, 12)
            else:
                period = min(len(series_clean) // 2, 12)

        if len(series_clean) < 2 * period:
            period = max(2, len(series_clean) // 3)

        # Try both additive & multiplicative, pick best
        best_model, best_result, best_residuals, best_var = None, None, None, np.inf
        for model in ["additive", "multiplicative"]:
            try:
                result = seasonal_decompose(series_clean, model=model, period=period, extrapolate_trend='freq')
                residuals = result.resid.dropna()
                if len(residuals) > 0:
                    var_res = np.var(residuals)
                    if var_res < best_var:
                        best_var = var_res
                        best_model = model
                        best_result = result
                        best_residuals = residuals
            except Exception:
                continue

        if best_result is None:
            return [], None, None, None

        # IQR threshold
        Q1, Q3 = best_residuals.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        outlier_mask = (best_residuals < lower_bound) | (best_residuals > upper_bound)
        outlier_indices = best_residuals.index[outlier_mask].tolist()

        return outlier_indices, best_result, best_residuals, best_model

    except Exception as e:
        st.error(f"Seasonal decomposition failed for '{series.name}': {e}")
        return [], None, None, None

# -----------------------------
# Helpers for Step 3: Stationarity
# -----------------------------
def adf_summary(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 10:
        return {"ok": False, "msg": "Not enough data for ADF (<10 points)."}
    stat = adfuller(s, autolag="AIC")
    return {
        "ok": True,
        "statistic": stat[0],
        "pvalue": stat[1],
        "lags_used": stat[2],
        "nobs": stat[3],
        "crit": stat[4],
        "is_stationary": stat[1] < 0.05
    }

def linear_detrend_with_trend(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    s = series.dropna()
    t = np.arange(len(s), dtype=float)
    m, b = np.polyfit(t, s.values.astype(float), 1)
    trend = m * t + b
    detr = pd.Series(s.values - trend, index=s.index, name=s.name)
    return detr.reindex(series.index), trend

# -----------------------------
# Helpers for Step 4: Rolling Average + Stats
# -----------------------------
def rolling_average(series, window=3):
    series_pd = pd.Series(series)
    smoothed = series_pd.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed.values

def window_stats(raw, smoothed, dates, window=5):
    raw_series = pd.Series(raw, index=dates)
    smoothed_series = pd.Series(smoothed, index=dates)

    stats = []
    for i in range(0, len(raw) - window + 1):
        raw_window = raw_series.iloc[i:i+window]
        smooth_window = smoothed_series.iloc[i:i+window]

        raw_std = raw_window.std()
        smooth_std = smooth_window.std()
        raw_var = raw_window.var()
        smooth_var = smooth_window.var()

        std_abs_diff = abs(raw_std - smooth_std)
        var_abs_diff = abs(raw_var - smooth_var)

        stats.append({
            "Window_Start_Date": raw_window.index[0],
            "Window_End_Date": raw_window.index[-1],
            "Raw_Mean": raw_window.mean(),
            "Smoothed_Mean": smooth_window.mean(),
            "Raw_Std": raw_std,
            "Smoothed_Std": smooth_std,
            "AbsDiff_Std": std_abs_diff,
            "Raw_Var": raw_var,
            "Smoothed_Var": smooth_var,
            "AbsDiff_Var": var_abs_diff,
            "Raw_Min": raw_window.min(),
            "Smoothed_Min": smooth_window.min(),
            "Raw_Max": raw_window.max(),
            "Smoothed_Max": smooth_window.max()
        })
    return pd.DataFrame(stats)

# -----------------------------
# Streamlit App (Pipeline)
# -----------------------------
st.title("📈 Time Series Pipeline: Residual-based Outlier Detection")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Preview initial raw data
    st.subheader("Preview of Initial Raw Data")
    st.dataframe(df.head(10))

    # Step 1: Sampling
    st.subheader("Step 1️⃣ - Sampling")
    time_col = st.selectbox("Time column", df.columns, index=0)
    numeric_cols = [c for c in df.columns if c != time_col]
    value_col = st.selectbox("Value column", numeric_cols, index=0)

    series_raw = ensure_time_series(df, time_col, value_col)
    series_resampled, rule = auto_resample(series_raw)

    st.line_chart(series_resampled, height=250)

    # Step 2: Outlier Detection (Residuals only)
    st.subheader("Step 2️⃣ - Outlier Detection (Residuals only)")
    outlier_indices, decomposition_result, residuals, chosen_model = detect_outliers_residuals(series_resampled)

    if chosen_model is not None:
        st.success(f"✅ Best model chosen automatically: **{chosen_model.capitalize()}** (lower residual variance)")

    if decomposition_result is not None:
        st.write("**Seasonal Decomposition Components:**")
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition_result.observed.plot(ax=axes[0], title='Original')
        decomposition_result.trend.plot(ax=axes[1], title='Trend')
        decomposition_result.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition_result.resid.plot(ax=axes[3], title='Residuals')
        plt.tight_layout()
        st.pyplot(fig)

    # Display outlier detection results
    if outlier_indices:
        outlier_mask = series_resampled.index.isin(outlier_indices)
        num_outliers = len(outlier_indices)

        st.write(f"**Outliers detected:** {num_outliers} out of {len(series_resampled)} points ({num_outliers/len(series_resampled)*100:.1f}%)")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series_resampled.index, series_resampled.values, label="Original Series", alpha=0.7, linewidth=1)
        ax.scatter(series_resampled.index[outlier_mask], series_resampled.values[outlier_mask],
                   color='red', s=50, label=f'Outliers ({num_outliers})', zorder=5)
        
        ax.set_title("Outlier Detection Results - Residuals")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        treat_outliers_choice = st.radio(
            "Do you want to treat the outliers?",
            ["No", "Yes"]
        )

        if treat_outliers_choice == "No":
            st.info("Pipeline stopped. No further processing will be done.")
            st.stop()
        else:
            final_series_for_processing = series_resampled
    else:
        st.success("No outliers detected!")
        continue_choice = st.radio(
            "No outliers found. Do you want to continue with the pipeline?",
            ["No", "Yes"]
        )
        if continue_choice == "No":
            st.info("Pipeline stopped.")
            st.stop()
        final_series_for_processing = series_resampled

    # Step 3: Stationarity processing
    st.subheader("Step 3️⃣ - Stationarity Processing")
    adf_res = adf_summary(final_series_for_processing)
    final_series = final_series_for_processing.copy()
    trend = None

    if adf_res["ok"]:
        if not adf_res["is_stationary"]:
            st.warning("❌ Series is NOT stationary → applying linear detrending…")
            detr, trend = linear_detrend_with_trend(final_series_for_processing)
            st.line_chart(detr)
            final_series = detr
        else:
            st.success("✅ Series is stationary!")
    else:
        st.warning(adf_res["msg"])

    # Step 4: Rolling Average + Stats
    st.subheader("Step 4️⃣ - Rolling Average + Statistics")
    dates = final_series.index
    series = final_series.values
    window_size = st.slider("Rolling / Stats Window Size", 3, 15, 3, step=2)
    smoothed_series = rolling_average(series, window=window_size)

    if trend is not None:
        if len(trend) != len(final_series):
            t = np.arange(len(final_series), dtype=float)
            original_series = final_series_for_processing.dropna()
            t_orig = np.arange(len(original_series), dtype=float)
            m, b = np.polyfit(t_orig, original_series.values.astype(float), 1)
            trend = m * t + b
        smoothed_series = smoothed_series + trend

    has_time = df[time_col].astype(str).str.contains(":").any()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, series + (trend if trend is not None else 0), label="Processed Series", alpha=0.5)
    ax.plot(dates, smoothed_series, label=f"Smoothed (w={window_size})", color="orange", linewidth=2)
    ax.set_title("Final Processed vs Smoothed Series")
    ax.set_xlabel("Date" if not has_time else "Timestamp")
    ax.set_ylabel("Value")
    ax.legend()
    if not has_time:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    else:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    stats_df = window_stats(series + (trend if trend is not None else 0), smoothed_series, dates, window=window_size)
    st.subheader("Window-wise Statistics")
    st.dataframe(stats_df)

    if has_time:
        smoothed_df_final = pd.DataFrame({
            "Timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "Smoothed_Value": smoothed_series
        })
    else:
        smoothed_df_final = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Smoothed_Value": smoothed_series
        })

    st.subheader("📥 Final Smoothed Data")
    st.dataframe(smoothed_df_final)

    st.download_button("Download Smoothed Data",
                       smoothed_df_final.to_csv(index=False).encode("utf-8"),
                       "smoothed_data_final.csv", "text/csv")
