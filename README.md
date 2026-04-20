# 🚀 Categorical & Numerical Data EDA + AutoML

This module focuses on structured datasets (categorical & numerical) using a low-code pipeline that integrates LLM-based feature selection with automated machine learning.

---

## 📌 Workflow

```text
Full Dataset + Problem Statement
            ↓
Chatbot Interface (chatbot_app.py)
            ↓
LLM Processing (llm_eda.py)
→ Identifies target variable
→ Selects relevant features
→ Filters dataset
            ↓
Optimized Dataset
            ↓
EDA & Preprocessing (app.py + module.py)
→ Data cleaning
→ Missing Data Handelling
→ Outlier Detecting
→ Encoding
→ Feature scaling
→ Statistical analysis and Visualisation
            ↓
Processed Dataset
            ↓
AutoML Pipeline (apps.py + modules.py)  (in progress)
→ Train multiple models  (in progress)
→ Compare performance  (in progress)
→ Prediction & evaluation  (in progress)
→ Visualization (in progress)
```

---

## ⚙️ Features

* LLM-based feature & target selection
* Automated preprocessing pipeline
* Exploratory Data Analysis (EDA)
* AutoML model training
* Prediction & evaluation

---

## 📂 Project Structure

```
Low-Code-Framework/
│
├── app.py                # EDA & preprocessing UI
├── apps.py               # AutoML pipeline UI (in progress)
├── chatbot_app.py        # Chatbot interface
├── llm_eda.py            # LLM-based feature/target selection
├── module.py             # Data preprocessing utilities
├── modules.py            # ML models (in progress)
├── decisiontree.py       # Standalone ML script
│
├── datasets/
│   └── weather_dataset.csv
```

---

## ⚠️ Status

* AutoML improvements → In progress
* Visualization → In progress

---

## 🎯 Goal

To build an end-to-end low-code ML system for structured datasets using LLM-driven automation.


---

## 👩‍💻 Author

Chandrima Das

---

## 📄 License

This project is proprietary and confidential. Unauthorized use, copying, or distribution is strictly prohibited.
Internal / proprietary — Calsoft Pvt Ltd.
