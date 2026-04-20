
# 🚀 Time Series EDA, Visualization & Forecasting Framework

An **end-to-end low-code framework** for performing **Time Series Exploratory Data Analysis (EDA), preprocessing, and forecasting** using both **classical statistical methods and deep learning models**.

This project is built using **Streamlit**, enabling users to interactively upload data, clean it, and generate forecasts without writing code.

---

## 🔍 Core Pipeline

This project follows a **structured two-stage workflow**:

```
Time Series Dataset
        ↓
EDA & Preprocessing (Time_series_EDA.py)
        ↓
Cleaned Time Series Data (cleaned_timeseries.csv)
        ↓
Forecasting Models (Forecasting_models.py)
        ↓
Predictions & Visualization
```

👉 **Key Idea:**

* `sample_timeseries.csv` → Input to EDA
* `cleaned_timeseries.csv` → Output of EDA
* Cleaned data is used as **input for forecasting models**

---

## 📌 Features

### 🔹 1. Automated Time Series EDA

* Time column parsing & sorting
* Automatic frequency detection (Daily / Weekly / Monthly)
* Resampling & missing value handling
* Residual-based **outlier detection using seasonal decomposition**
* Stationarity check using **ADF test**
* Automatic detrending for non-stationary data
* Rolling average smoothing
* Window-based statistical analysis

---

### 🔹 2. Forecasting Models

#### 📊 Classical Models

* Rolling EWMA
* Holt-Winters (Triple Exponential Smoothing)
* ARIMA (Auto-tuned using AIC)
* SARIMA
* SARIMAX (with exogenous variables)
* VAR (Multivariate forecasting)

#### 🤖 Deep Learning Models

* RNN
* LSTM
* GRU
* TCN (Temporal Convolutional Network)

#### 🚀 Advanced (Planned)

* Seq2Seq Encoder–Decoder
* Transformer-based models (Chronos / TFT / Informer)

---

### 🔹 3. Evaluation & Visualization

* Train-test split (last 20%)
* MAE-based validation
* Observed vs Forecast plots
* Multi-step recursive forecasting

---

### 🔹 4. Interactive Streamlit Apps

* Upload CSV data
* Perform EDA & preprocessing
* Download cleaned dataset
* Run forecasting models
* Visualize predictions

---

## 📂 Project Structure

```text
Low-Code-Framework/
│
├── sample_timeseries.csv        # Raw/sample time series dataset used as input for the EDA pipeline
├── cleaned_timeseries.csv      # Cleaned and processed output from EDA, used as input for forecasting models
├── Time_series_EDA.py          # Streamlit app for time series preprocessing (resampling, outlier detection, stationarity, smoothing)
├── Forecasting_models.py       # Streamlit app implementing forecasting models (ARIMA, LSTM, GRU, etc.) on cleaned data
├── requirements.txt            # List of required Python dependencies to run the project
└── README.md                   # Project documentation with workflow, features, and usage instructions
```

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas, NumPy**
* **Matplotlib**
* **Statsmodels**
* **TensorFlow / Keras**

---

## ⚙️ Installation

```bash
git clone https://github.com/Chandrimadas2005/Low-Code-Framework.git
cd Low-Code-Framework
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Step 1: Run EDA Module

```bash
streamlit run Time_series_EDA.py
```

* Upload `sample_timeseries.csv`
* Perform preprocessing
* Download `cleaned_timeseries.csv`

---

### 🔹 Step 2: Run Forecasting Module

```bash
streamlit run Forecasting_models.py
```

* Upload the **cleaned dataset**
* Select forecasting model
* View predictions and evaluation

---

## 📊 Input Format

### For EDA:

```text
Date, Value
2020-01-01, 100
2020-01-02, 120
```

### For Forecasting:

* Use the **cleaned output generated from EDA**

Optional:

```text
Date, Value, Exogenous1, Exogenous2
```

---

## 💡 Highlights

* ✅ Complete pipeline: **Raw Data → Cleaned Data → Forecasting**
* ✅ Automated preprocessing & statistical validation
* ✅ Residual-based outlier detection (advanced approach)
* ✅ Supports both statistical and deep learning models
* ✅ Low-code, interactive UI
* ✅ Real-world ready workflow

---

## 🚀 Future Improvements

* Model comparison dashboard (MAE, RMSE, MAPE)
* Feature engineering (lag features, rolling stats, time features)
* Confidence intervals for forecasts
* Integration with Prophet, XGBoost
* Transformer-based forecasting (Chronos, TFT)
* Deployment (Streamlit Cloud / Docker)

---

## 📌 Use Cases

* Sales forecasting
* Demand prediction
* Financial time series analysis
* Sensor data monitoring
* Industrial analytics

---

## 👩‍💻 Contributor

**Bratati Basu**

---

