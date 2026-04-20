import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ======================
# 🔹 Step 1: Load Dataset
# ======================
file_path = "your_dataset.csv"   # 👈 Replace with your file
data = pd.read_csv(file_path)

print("Dataset loaded successfully ✅")
print("Shape:", data.shape)
print("Columns:", data.columns)

# ===========================
# 🔹 Step 2: Define Features & Target
# ===========================
target_col = input("Enter target column name: ")  # user specifies target
X = data.drop(columns=[target_col])
y = data[target_col]

# ===========================
# 🔹 Step 3: Decide Task Type
# ===========================
# If target is numeric → Regression
# If target is categorical (string/object or few unique values) → Classification
task_type = "regression"

if y.dtype == "object" or y.nunique() < 20:  # categorical
    task_type = "classification"

print(f"Task detected: {task_type.upper()}")

# Encode categorical target for classification
if task_type == "classification" and y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ===========================
# 🔹 Step 4: Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ===========================
# 🔹 Step 5: Train Model
# ===========================
if task_type == "regression":
    model = DecisionTreeRegressor(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("📉 Regression Results:")
    print("MSE:", mse)

else:  # classification
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("📊 Classification Results:")
    print("Accuracy:", acc)

# ===========================
# 🔹 Step 6: Show Predictions
# ===========================
print("\nSample Predictions:")
print(pd.DataFrame({"Actual": y_test[:10], "Predicted": y_pred[:10]}))