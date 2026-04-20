import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, Flatten
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
import warnings

warnings.filterwarnings("ignore")

st.title("Classical Time Series Forecasting App")

# Inputs
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Value[, Exogenous...])", type=["csv"])

method_list = [
    "Rolling EWMA (window-based)",
    "Holt-Winters (Triple Exp. Smoothing)",
    "ARIMA",
    "SARIMA",
    "SARIMAX",
    "VAR (Vector AutoRegression)",
    "RNN",
    "LSTM",
    "GRU",
    "TCN (Temporal Convolutional Network)",
    "Seq2Seq Encoder–Decoder",
    "Transformer (TFT / Informer / Chronos)"
]
method = st.sidebar.selectbox("Select Forecasting Method", method_list)
forecast_periods = st.sidebar.number_input("Periods to Forecast", min_value=1, max_value=60, value=10)

def select_arima_order(ts, seasonal=False, m=1, exog=None):
    best_aic = np.inf
    best_order = None
    best_seasonal = None
    max_p, max_d, max_q = 2, 1, 2
    max_P, max_D, max_Q = 1, 1, 1

    for p in range(max_p+1):
        for d in range(max_d+1):
            for q in range(max_q+1):
                if seasonal:
                    for P in range(max_P+1):
                        for D in range(max_D+1):
                            for Q in range(max_Q+1):
                                try:
                                    model = SARIMAX(ts, exog=exog,
                                                    order=(p,d,q),
                                                    seasonal_order=(P,D,Q,m),
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                                    res = model.fit(disp=False)
                                    if res.aic < best_aic:
                                        best_aic = res.aic
                                        best_order = (p,d,q)
                                        best_seasonal = (P,D,Q,m)
                                except Exception:
                                    continue
                else:
                    try:
                        model = ARIMA(ts, order=(p,d,q), exog=exog)
                        res = model.fit()
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (p,d,q)
                    except Exception:
                        continue
    return best_order, best_seasonal

forecast = None
ts = None
train_ts = None
test_ts = None
future_idx = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    value_columns = st.sidebar.multiselect("Select value columns for forecasting", df.columns.tolist(), default=df.columns[:1].tolist())

    exog_columns = []
    if method == "SARIMAX":
        exog_columns = st.sidebar.multiselect("Select exogenous columns (optional)", df.columns.tolist(), default=[])
        exog_train = df[exog_columns] if exog_columns else None
    else:
        exog_train = None

    if method == "VAR (Vector AutoRegression)":
        if len(value_columns) < 2:
            st.warning("VAR requires at least two value columns. Please select two or more columns.")
            ts = None
        else:
            ts = df[value_columns]
    else:
        ts = df[value_columns[0]]

    st.write("Raw data:")
    if ts is not None:
        st.line_chart(ts)

    freq = pd.infer_freq(ts.index)
    if freq is None:
        freq = pd.date_range(ts.index[0], ts.index[-1]).freqstr or "D"
    if "M" in freq:
        sp = 12
    elif "Q" in freq:
        sp = 4
    elif "W" in freq:
        sp = 52
    elif "D" in freq:
        sp = 7
    else:
        sp = 12

    future_idx = pd.date_range(ts.index[-1], periods=forecast_periods+1, freq=freq)[1:]

    test_size = min(10, int(len(ts) * 0.2))
    train_ts = ts.iloc[:-test_size] if test_size > 0 else ts
    test_ts = ts.iloc[-test_size:] if test_size > 0 else None

    # Classical models
    if method == "Rolling EWMA (window-based)":
        ewma_window = st.sidebar.number_input("EWMA window", min_value=2, max_value=60, value=5)
        ewma_alpha = st.sidebar.slider("EWMA alpha", min_value=0.01, max_value=1.0, value=0.2)
        history = list(train_ts.values)
        forecast_vals = []
        for i in range(forecast_periods):
            window = pd.Series(history[-ewma_window:])
            ewma_val = window.ewm(alpha=ewma_alpha, adjust=False).mean().iloc[-1]
            forecast_vals.append(ewma_val)
            history.append(ewma_val)
        forecast = pd.Series(forecast_vals, index=future_idx)

    elif method == "Holt-Winters (Triple Exp. Smoothing)":
        alpha = st.sidebar.slider("HW alpha", min_value=0.01, max_value=1.0, value=0.2)
        beta = st.sidebar.slider("HW beta", min_value=0.01, max_value=1.0, value=0.1)
        gamma = st.sidebar.slider("HW gamma", min_value=0.01, max_value=1.0, value=0.1)
        model = ExponentialSmoothing(train_ts, trend='add', seasonal='add', seasonal_periods=sp).fit(
            smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma, optimized=True)
        forecast = model.forecast(forecast_periods)
        forecast.index = future_idx

    elif method == "ARIMA":
        best_order, _ = select_arima_order(train_ts, seasonal=False, m=sp)
        if best_order is None:
            best_order = (1,1,1)
        model = ARIMA(train_ts, order=best_order).fit()
        forecast = model.forecast(forecast_periods)
        forecast.index = future_idx

    elif method == "SARIMA":
        with st.spinner("Fitting SARIMA model, please wait..."):
            best_order, best_seasonal = select_arima_order(train_ts, seasonal=True, m=sp)
            if best_order is None:
                best_order = (1,1,1)
            if best_seasonal is None:
                best_seasonal = (1,1,1,sp)
            model = SARIMAX(train_ts, order=best_order, seasonal_order=best_seasonal).fit()
            forecast = model.forecast(forecast_periods)
            forecast.index = future_idx

    elif method == "SARIMAX":
        with st.spinner("Fitting SARIMAX model, please wait..."):
            best_order, best_seasonal = select_arima_order(train_ts, seasonal=True, m=sp, exog=exog_train)
            if best_order is None:
                best_order = (1,1,1)
            if best_seasonal is None:
                best_seasonal = (1,1,1,sp)
            model = SARIMAX(train_ts, order=best_order, seasonal_order=best_seasonal, exog=exog_train).fit()
            forecast = model.forecast(forecast_periods, exog=exog_train.tail(forecast_periods) if exog_train is not None else None)
            forecast.index = future_idx

    elif method == "VAR (Vector AutoRegression)":
        if ts is None:
            forecast = None
        else:
            try:
                model = VAR(ts)
                order_results = model.select_order(maxlags=min(10, len(ts)//2))
                lag_order = order_results.aic if hasattr(order_results, 'aic') and order_results.aic is not None else 1
                if isinstance(lag_order, (np.ndarray, list)):
                    lag_order = int(np.nanargmin(lag_order))
                else:
                    lag_order = int(lag_order)
                model_fit = model.fit(lag_order)
                forecast_values = model_fit.forecast(ts.values[-lag_order:], forecast_periods)
                forecast = pd.DataFrame(forecast_values, index=future_idx, columns=ts.columns)
            except Exception as e:
                st.error(f"VAR model error: {e}")
                forecast = None

    # Deep learning models
    elif method in ["RNN", "LSTM", "GRU", "TCN (Temporal Convolutional Network)"]:
        n_input = st.sidebar.slider("Input window size", 5, 50, 10)
        n_neurons = st.sidebar.slider("Hidden neurons", 10, 200, 50)
        epochs = st.sidebar.slider("Training epochs", 5, 50, 20)

        gen = TimeseriesGenerator(train_ts.values, train_ts.values, length=n_input, batch_size=1)
        model = Sequential()

        if method == "RNN":
            model.add(SimpleRNN(n_neurons, activation="tanh", input_shape=(n_input,1)))
        elif method == "LSTM":
            model.add(LSTM(n_neurons, activation="tanh", input_shape=(n_input,1)))
        elif method == "GRU":
            model.add(GRU(n_neurons, activation="tanh", input_shape=(n_input,1)))
        elif method == "TCN (Temporal Convolutional Network)":
            model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(n_input,1)))
            model.add(Flatten())

        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(gen, epochs=epochs, verbose=0)

        history = list(train_ts.values)
        forecast_vals = []
        for _ in range(forecast_periods):
            x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
            yhat = model.predict(x_input, verbose=0)
            forecast_vals.append(yhat[0][0])
            history.append(yhat[0][0])
        forecast = pd.Series(forecast_vals, index=future_idx)

    elif method == "Seq2Seq Encoder–Decoder":
        st.info("Seq2Seq demo placeholder: build encoder-decoder LSTM for multi-step forecasting.")
        forecast = pd.Series([np.nan]*forecast_periods, index=future_idx)

    elif method == "Transformer (TFT / Informer / Chronos)":
        st.info("Transformer-based forecasting requires HuggingFace models (Chronos, TimeGPT).")
        st.write("Example: Load Chronos model from HuggingFace for pretrained forecasting.")
        forecast = pd.Series([np.nan]*forecast_periods, index=future_idx)

    # Display forecast
    st.subheader("Forecasted values")
    if forecast is not None:
        st.write(forecast)

    # Validation metrics
    if test_ts is not None:
        st.subheader("Validation (last {} points)".format(test_size))
        if method == "VAR (Vector AutoRegression)":
            if ts is not None:
                try:
                    pred = model_fit.forecast(ts.values[-lag_order:], test_size)
                    pred_df = pd.DataFrame(pred, index=test_ts.index, columns=ts.columns)
                    for col in ts.columns:
                        mae = np.mean(np.abs(test_ts[col] - pred_df[col]))
                        st.write(f"{col} MAE: {mae:.3f}")
                except Exception as e:
                    st.write(f"Validation error: {e}")
        else:
            pred = None
            try:
                if method == "Rolling EWMA (window-based)":
                    history = list(train_ts.values)
                    pred_vals = []
                    for i in range(test_size):
                        window = pd.Series(history[-ewma_window:])
                        ewma_val = window.ewm(alpha=ewma_alpha, adjust=False).mean().iloc[-1]
                        pred_vals.append(ewma_val)
                        history.append(test_ts.values[i])
                    pred = np.array(pred_vals)
                elif method == "Holt-Winters (Triple Exp. Smoothing)":
                    pred = model.forecast(test_size)
                elif method == "ARIMA":
                    pred = model.forecast(test_size)
                elif method == "SARIMA":
                    pred = model.forecast(test_size)
                elif method == "SARIMAX":
                    pred = model.forecast(test_size)
                elif method in ["RNN", "LSTM", "GRU", "TCN (Temporal Convolutional Network)"]:
                    history = list(train_ts.values)
                    pred_vals = []
                    for i in range(test_size):
                        x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
                        yhat = model.predict(x_input, verbose=0)
                        pred_vals.append(yhat[0][0])
                        history.append(test_ts.values[i])
                    pred = np.array(pred_vals)
                if pred is not None:
                    mae = np.mean(np.abs(test_ts.values - pred))
                    st.write(f"MAE: {mae:.3f}")
            except Exception as e:
                st.write(f"Validation error: {e}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    if method == "VAR (Vector AutoRegression)":
        if ts is not None and forecast is not None:
            for col in ts.columns:
                ax.plot(ts[col], label=f"Observed {col}")
                ax.plot(forecast[col], label=f"Forecast {col}")
    else:
        ax.plot(ts, label="Observed", color="blue")
        if forecast is not None:
            ax.plot(pd.concat([pd.Series(ts.iloc[-1], index=[ts.index[-1]]), forecast]), label="Forecast", color="red")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Upload a CSV with columns: Date, Value[, other columns as needed]")