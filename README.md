
# 🚀 Time Series EDA, Visualization & Forecasting Framework

An **end-to-end low-code framework** for performing **Time Series Exploratory Data Analysis (EDA), preprocessing, and forecasting** using both **classical statistical methods and deep learning models**.

Built with an interactive **Streamlit UI**, this project enables users to upload datasets, clean and analyze time series data, and generate forecasts — all without writing code.

---

## 📌 Features

### 🔍 1. Automated Time Series EDA Pipeline

* Time column parsing & sorting
* Automatic frequency detection (Daily / Weekly / Monthly)
* Resampling & missing value handling (interpolation, forward/backward fill)
* Residual-based **outlier detection using seasonal decomposition**
* Stationarity check using **ADF test**
* Automatic detrending for non-stationary data
* Rolling average smoothing
* Window-based statistical analysis (mean, variance, std deviation)

---

### 📈 2. Forecasting Models

#### 🔹 Classical Models

* Rolling EWMA
* Holt-Winters (Triple Exponential Smoothing)
* ARIMA (Auto-tuned using AIC)
* SARIMA
* SARIMAX (with exogenous variables)
* VAR (Multivariate forecasting)

#### 🔹 Deep Learning Models

* RNN
* LSTM
* GRU
* TCN (Temporal Convolutional Network)

#### 🔹 Advanced (Planned)

* Seq2Seq Encoder–Decoder
* Transformer-based models (Chronos / TFT / Informer)

---

### 📉 3. Evaluation & Visualization

* Train-test split (last 20%)
* MAE-based validation
* Observed vs Forecast plots
* Multi-step recursive forecasting

---

### 🖥️ 4. Interactive Streamlit UI

* Upload CSV file (`Date`, `Value`, optional exogenous variables)
* Select forecasting model
* Tune hyperparameters
* Visualize raw, processed, and forecasted data
* Download cleaned dataset

---

## 🔄 Workflow

```text
Raw Time Series Data
        ↓
EDA & Preprocessing Pipeline
        ↓
Cleaned Time Series
        ↓
Forecasting Models
        ↓
Predictions + Evaluation + Visualization
```

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas, NumPy
* Matplotlib
* Statsmodels
* TensorFlow / Keras

---

## ⚙️ Installation

```bash
git clone https://github.com/Chandrimadas2005/Low-Code-Framework.git
cd Low-Code-Framework
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### Run EDA Module

```bash
streamlit run Time_series_EDA.py
```

### Run Forecasting Module

```bash
streamlit run Forecasting_models.py
```

---

## 📊 Input Format

CSV file should contain:

```text
Date, Value
2020-01-01, 100
2020-01-02, 120
...
```

Optional:

```text
Date, Value, Exogenous1, Exogenous2
```

---

## 💡 Highlights

* ✅ End-to-end pipeline (EDA → preprocessing → forecasting)
* ✅ Automated data cleaning & statistical validation
* ✅ Supports both statistical and deep learning models
* ✅ Low-code, user-friendly interface
* ✅ Scalable for real-world time series problems

---

## 🚀 Future Improvements

* Model comparison dashboard (MAE, RMSE, MAPE)
* Feature engineering (lag features, rolling statistics, time features)
* Confidence intervals for forecasts
* Integration with Prophet, XGBoost
* Transformer-based forecasting (Chronos, TFT)
* Deployment (Streamlit Cloud / Docker)

---

## 📌 Use Cases

* Sales & demand forecasting
* Financial time series analysis
* Sensor data monitoring
* Industrial analytics
* Business trend prediction

---

## 👩‍💻 Contributor

**Bratati Basu**
---

