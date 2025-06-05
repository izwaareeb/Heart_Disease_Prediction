import streamlit as st
import numpy as np
import joblib
import pandas as pd
from streamlit_lottie import st_lottie
import requests
import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        st.info("🎙️ Listening... Please speak now.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"✅ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("😕 Sorry, I couldn't understand the audio.")
    except sr.RequestError:
        st.error("❌ Could not request results from the speech recognition service.")
    return ""

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "This app predicts your risk of heart disease based on input data."
    }
)

# ------------------- Load model & scaler -------------------
model = joblib.load("D:/Project/NEW PROJECT/heart_disease_prediction/heart_model.pkl")
scaler = joblib.load("D:/Project/NEW PROJECT/heart_disease_prediction/scaler.pkl")

# ------------------- Background Setup -------------------
default_bg = """
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, #222831, #000000);
    overflow: hidden;
}
.animated-bg {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: -1;
    background: url("https://cdn.pixabay.com/photo/2023/07/17/19/39/ai-generated-8132343_1280.png") center/cover no-repeat;
    opacity: 0.08;
    animation: pulse 10s infinite alternate;
}
@keyframes pulse {
    0% { transform: scale(1); }
    100% { transform: scale(1.05); }
}
</style>
<div class="animated-bg"></div>
"""
st.markdown(default_bg, unsafe_allow_html=True)

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
}
.login-btn {
    background: linear-gradient(90deg, #00f260, #0575e6);
    border: none;
    color: white;
    padding: 12px 30px;
    font-size: 18px;
    border-radius: 30px;
    cursor: pointer;
    font-weight: bold;
    margin-top: 20px;
    box-shadow: 0 4px 15px rgba(0, 242, 96, 0.6);
}
.login-btn:hover {
    background: linear-gradient(90deg, #0575e6, #00f260);
    box-shadow: 0 6px 20px rgba(5, 117, 230, 0.8);
}
.success-msg { color: #28a745; font-weight: bold; margin: 10px 0; }
.error-msg { color: #dc3545; font-weight: bold; margin: 10px 0; }
.info-msg { color: #17a2b8; font-weight: bold; margin: 10px 0; }
.main-container { padding: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ------------------- Session State Init -------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# ------------------- Login Section -------------------
if not st.session_state["logged_in"]:
    st.header("🔐 Secure Access")
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")
    if st.button("Login", key="login_btn"):
    st.session_state["logged_in"] = True
    st.rerun()

else:
    if st.button("Logout", key="logout_btn"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.markdown('<p class="success-msg">Welcome! 🎉</p>', unsafe_allow_html=True)

    def load_lottieurl(url):
        try:
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        except:
            return None

    def validate_inputs(age, totChol, sysBP, glucose):
        errors = []
        if not (20 <= age <= 100): errors.append("Age must be between 20 and 100.")
        if not (100 <= totChol <= 400): errors.append("Cholesterol must be between 100 and 400.")
        if not (90 <= sysBP <= 250): errors.append("Systolic BP must be between 90 and 250.")
        if not (50 <= glucose <= 300): errors.append("Glucose must be between 50 and 300.")
        return errors

    # ------------------- Input Form -------------------
    st.header("❤️ Heart Disease Risk Predictor")
    st.markdown("Get insights about your heart health based on clinical data. Powered by Logistic Regression.")
    st.markdown("---")

    st.subheader("🔎 Input Your Health Data")

    st.markdown("### 👤 Personal Information")
    col1, col2 = st.columns([1, 2])

    with col1:
        col_age1, col_age2 = st.columns([3, 1])
        with col_age1:
            age = st.number_input("🧓 Age", 20, 100, 50, key="age_input")
        with col_age2:
            if st.button("🎤 Speak Age"):
                spoken_text = recognize_speech()
                if spoken_text.isdigit():
                    st.session_state.age_input = int(spoken_text)
                else:
                    st.error("⚠️ Please speak a valid number for age.")

        gender = st.selectbox("Gender", ("Male", "Female"))

    with col2:
        family_history = st.radio("Family History of Heart Disease?", ("Yes", "No"))
        family_member = st.text_input("🧬 Who in your family had heart disease?") if family_history == "Yes" else ""

    st.markdown("### 📋 Clinical Measurements")
    col3, col4 = st.columns(2)

    with col3:
        col_chol1, col_chol2 = st.columns([3, 1])
        with col_chol1:
            totChol = st.number_input("🧪 Total Cholesterol", 100, 400, 200, key="chol_input")
        with col_chol2:
            if st.button("🎤 Speak Cholesterol"):
                spoken_text = recognize_speech()
                if spoken_text.isdigit():
                    st.session_state.chol_input = int(spoken_text)
                else:
                    st.error("⚠️ Please speak a valid number for cholesterol.")

        glucose = st.number_input("🩸 Glucose Level", 50, 300, 100)

    with col4:
        sysBP = st.number_input("💉 Systolic Blood Pressure", 90, 250, 120)

    st.markdown("### 🔍 Prediction")

    if st.button("Predict", key="predict_btn"):
        errors = validate_inputs(age, totChol, sysBP, glucose)
        if errors:
            for error in errors:
                st.markdown(f'<p class="error-msg">{error}</p>', unsafe_allow_html=True)
        else:
            with st.spinner('Predicting...'):
                input_data = np.array([[age, 1 if gender == "Male" else 0, totChol, sysBP, glucose, 1 if family_history == "Yes" else 0]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0][1]

                # Background based on risk
                if prediction_proba < 0.3:
                    bg_color = "#d4edda"  # green
                    risk_label = "Low Risk"
                elif prediction_proba < 0.7:
                    bg_color = "#fff3cd"  # yellow
                    risk_label = "Medium Risk"
                else:
                    bg_color = "#f8d7da"  # red
                    risk_label = "High Risk"

                st.markdown(f"""
                <style>
                    .stApp {{
                        background-color: {bg_color} !important;
                    }}
                </style>
                """, unsafe_allow_html=True)

                st.markdown(f'<p class="info-msg">Prediction Confidence: <strong>{prediction_proba:.2%}</strong></p>', unsafe_allow_html=True)
                st.progress(int(prediction_proba * 100))

                if prediction == 1:
                    st.error("⚠️ You may be at risk of heart disease. Please consult a healthcare professional.")
                else:
                    st.success("✅ You are unlikely to have heart disease currently.")

                st.session_state.prediction_history.append({
                    "Age": age,
                    "Gender": gender,
                    "Family History": family_history,
                    "Family Member": family_member if family_history == "Yes" else "N/A",
                    "Total Cholesterol": totChol,
                    "Glucose": glucose,
                    "Systolic BP": sysBP,
                    "Risk Probability": f"{prediction_proba:.2%}",
                    "Prediction": risk_label
                })

    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("🕒 Prediction History")
        df_history = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_history)

        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Report as CSV",
            data=csv,
            file_name='heart_disease_prediction_report.csv',
            mime='text/csv',
        )

    lottie_url = "https://assets8.lottiefiles.com/packages/lf20_8t8a5uuc.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, speed=1, width=300, height=300, key="heartbeat")

st.markdown('</div>', unsafe_allow_html=True)