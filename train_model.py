# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"D:\Project\NEW PROJECT\heart_disease_prediction\data\framingham.csv")
df.drop(['education'], axis=1, inplace=True)
df.rename(columns={'male': 'Sex_male'}, inplace=True)
df.dropna(inplace=True)

# Add synthetic FamilyHistory column (random 0 or 1)
np.random.seed(42)  # reproducibility
df['FamilyHistory'] = np.random.randint(0, 2, df.shape[0])

# Select new feature set
X = np.asarray(df[['age', 'Sex_male', 'totChol', 'sysBP', 'glucose', 'FamilyHistory']])
y = np.asarray(df['TenYearCHD'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=4)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Save model and scaler
joblib.dump(logreg, r"D:\Project\NEW PROJECT\heart_disease_prediction\heart_model.pkl")
joblib.dump(scaler, r"D:\Project\NEW PROJECT\heart_disease_prediction\scaler.pkl")

print("✅ Model and Scaler saved successfully with new features.")
