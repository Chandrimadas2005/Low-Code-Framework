import google.generativeai as genai
import pandas as pd
import json
import re

# Configure Gemini API
genai.configure(api_key="YOUR_API_KEY_HERE")

# Load your dataset
df = pd.read_csv("datasets/weather_dataset.csv")

# Take problem statement as chatbot-like input
user_problem = input("Please describe your problem: ")

# Build prompt
prompt = f"""
You are an expert data scientist.
 
The user’s problem is: "{user_problem}"
 
The dataset columns are: {list(df.columns)}
 
Your task:

1. Based on the problem, identify the most appropriate target column (only one).
2. Select the useful feature columns that can serve as predictors.
3. Return the result only in valid JSON format with this structure:

{{
  "target": "target_column_name",
  "features": ["list", "of", "features"]
}}
"""

# Use Gemini model
model = genai.GenerativeModel('YOUR_MODEL_NAME_HERE')
response = model.generate_content(prompt)

# Clean response text
clean_text = response.text.strip()
clean_text = re.sub(r"```(json)?", "", clean_text).strip()
clean_text = clean_text.replace("```", "").strip()

try:
    result = json.loads(clean_text)
    target_col = result["target"]
    feature_cols = result["features"]

    # ✅ Print target and features
    print("\n📌 Target Column:", target_col)
    print("📌 Feature Columns:", feature_cols)

    # ✅ Filter dataset
    df_selected = df[[target_col] + feature_cols]

    # ✅ Save new dataset
    output_path = r"datasets/weather_dataset.csv"
    df_selected.to_csv(output_path, index=False)
    print(f"\n✅ New dataset saved at: {output_path}")

except Exception as e:
    print("⚠️ Error parsing Gemini response:", e)
    print("Raw text was:\n", clean_text)
