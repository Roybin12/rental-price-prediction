import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

from assets_data_prep import prepare_data

# --- טוענים את הנתונים הגולמיים ---
df = pd.read_csv("C:\\Users\\USER001\\Desktop\\מטלות מבוא לתכנות רועי\\train.csv")

# --- הכנת הנתונים ---
df_prepared = prepare_data(df, mode='train')

# --- הפרדת משתנים ---
X = df_prepared.drop(columns=["price"])
y = df_prepared["price"]

# --- אימון מודל ElasticNet ---
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X, y)

# --- שמירת עמודות הפיצ'רים (לשימוש ב-FLASK) ---
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# --- שמירת המודל המאומן לקובץ ---
joblib.dump(model, "trained_model.pkl")

print("✅ אימון הסתיים. המודל והפיצ'רים נשמרו.")
