# 🚀 Low-Code ML & Smart EDA Framework

A low-code machine learning and exploratory data analysis (EDA) framework that combines traditional data processing with LLM-powered automation to simplify end-to-end ML workflows.

---

## 📌 Overview

This project enables users to:

* Upload datasets
* Provide a problem statement in natural language
* Automatically identify target & relevant features using LLM
* Perform data cleaning and exploratory data analysis
* Train machine learning models with minimal manual effort

---

## 🔄 Workflow (How it works)

```text
Full Dataset + Problem Statement
            ↓
Chatbot App (chatbot_app.py)
            ↓
LLM Processing (llm_eda.py)
→ Identifies target variable & important features
→ Filters required rows & columns
            ↓
New Optimized Dataset
            ↓
EDA & Preprocessing (app.py + module.py)
→ Data cleaning
→ Encoding & scaling
→ Statistical insights
            ↓
Processed Dataset
            ↓
AutoML Pipeline (app.py + modules.py)
→ Model training (in progress)
→ Model selection & evaluation
```

---

## ⚙️ Features

### 🤖 LLM-Powered Automation

* Identifies target variable automatically
* Selects relevant features
* Converts problem statements into ML tasks

### 🧹 Data Preprocessing

* Missing value handling
* Outlier detection
* Categorical encoding
* Feature scaling

### 📊 Smart EDA

* Dataset statistics
* Data profiling
* Insight generation

### ⚡ Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost / LightGBM / CatBoost
* Support Vector Machine
* Neural Networks

### 💬 Interactive UI

* Built with Streamlit
* Chat-based user interaction

---

## 🏗️ Project Structure

```bash
Low-Code-Framework/
│
├── app.py                # EDA & preprocessing UI
├── apps.py               # AutoML pipeline (in progress)
├── chatbot_app.py        # Chatbot interface
├── llm_eda.py            # LLM-based feature/target selection
├── module.py             # Data preprocessing utilities
├── modules.py            # ML model training
├── decisiontree.py       # Standalone ML script
│
├── datasets/
│   ├── weather_dataset.csv
│   ├── weather_dataset_full.csv
│   └── weather_dataset_missing.csv
```

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/Low-Code-Framework.git
cd Low-Code-Framework
pip install -r requirements.txt
```

---

## ▶️ Usage (Step-by-step Workflow)

### 1️⃣ Start Chatbot (Define Problem)

```bash
streamlit run chatbot_app.py
```

* Enter your dataset
* Provide a problem statement
* LLM selects target & features
* Generates a filtered dataset

---

### 2️⃣ Run EDA & Preprocessing

```bash
streamlit run app.py
```

* Load filtered dataset
* Perform cleaning, encoding, scaling
* Analyze dataset insights

---

### 3️⃣ Run AutoML Pipeline *(In Progress)*

```bash
streamlit run apps.py
```

* Train multiple ML models
* Compare performance
* Select best model

---

## 🔑 Configuration

```bash
export GEMINI_API_KEY="your_api_key"
```

---

## 📊 Use Cases

* Rapid ML prototyping
* Automated EDA for beginners
* Academic research experiments
* Low-code ML pipelines

---

## ⚠️ Notes

* AutoML pipeline is currently under development
* Model explainability (SHAP/LIME) can be added
* UI improvements can enhance usability

---

## 👩‍💻 Author

Chandrima Das

---

## 📄 License

This project is proprietary and confidential. Unauthorized use, copying, or distribution is strictly prohibited.
