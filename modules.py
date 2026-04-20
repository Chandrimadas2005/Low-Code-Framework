import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Define available models
CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Trees": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting (XGBoost)": xgb.XGBClassifier,
    "Gradient Boosting (LightGBM)": lgb.LGBMClassifier,
    "Gradient Boosting (CatBoost)": cb.CatBoostClassifier,
    "Support Vector Machine": SVC,
    "Neural Networks": MLPClassifier
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Random Forest Regressor": RandomForestRegressor,
    "Gradient Boosting (XGBoost)": xgb.XGBRegressor,
    "Gradient Boosting (LightGBM)": lgb.LGBMRegressor,
    "Gradient Boosting (CatBoost)": cb.CatBoostRegressor,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "ElasticNet Regression": ElasticNet,
    "Neural Networks": MLPRegressor
}

def train_and_evaluate(X, y, problem_type, selected_models):
    """Train and evaluate models dynamically based on problem type"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for model_name in selected_models:
        if problem_type == "Classification":
            model_class = CLASSIFICATION_MODELS[model_name]
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[model_name] = {"Accuracy": acc, "Report": report}

        elif problem_type == "Regression":
            model_class = REGRESSION_MODELS[model_name]
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = {"MSE": mse, "R2": r2}

    return results
