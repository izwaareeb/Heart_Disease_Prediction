# 💓 Heart Disease Prediction App

A machine learning-powered web application that predicts the 10-year risk of heart disease based on user input. The model is trained on the **Framingham Heart Study** dataset using **Logistic Regression** and is deployed using **Streamlit** for real-time health risk assessment.

---

## 🚀 Features

- 🧠 Predicts 10-year heart disease risk using logistic regression
- 📊 Trained on real-world clinical data (Framingham dataset)
- 💡 Adds a synthetic `FamilyHistory` feature for experimentation
- 🌐 Intuitive and interactive UI using Streamlit
- 📈 Visual outputs for insights and better interpretability

---

## 🧠 Machine Learning Model

- Model: `LogisticRegression` (from scikit-learn)
- Preprocessing: `StandardScaler` for feature normalization
- Evaluation: Accuracy score, confusion matrix, and classification report
- Model is saved as `heart_model.pkl` and used in real-time prediction

---

## 📊 Dataset

- **Source**: Framingham Heart Study
- Columns used: `age`, `Sex_male`, `totChol`, `sysBP`, `glucose`, `FamilyHistory`
- Target variable: `TenYearCHD` (0 = No CHD, 1 = CHD within 10 years)

---

## 🛠️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-predictor.git
   cd heart-disease-predictor
