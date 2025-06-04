import streamlit as st
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import requests

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to load Lottie animations safely
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# Original Lottie animations (may fail sometimes if LottieFiles updates or restricts)
lottie_health = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_success = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_t6shlqjz.json")
lottie_warning = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_hzgq1iov.json")

# Streamlit page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .title {
        color: #3E4E88;
        font-size: 32px;
        font-weight: 700;
        text-align: center;
        padding-top: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
if lottie_health:
    st_lottie(lottie_health, height=200, key="health")

st.markdown("### 🧾 Enter your medical details below to assess your diabetes risk:")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2927/2927541.png", width=120)
    st.markdown("## 🧠 About")
    st.info("This app uses a machine learning model to predict the risk of diabetes based on personal health metrics.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("👶 Pregnancies", min_value=0)
    glucose = st.number_input("🍬 Glucose Level", min_value=0)
    bp = st.number_input("🩸 Blood Pressure", min_value=0)
    skin = st.number_input("🧪 Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("💉 Insulin Level", min_value=0)
    bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=0.0)
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("🎂 Age", min_value=1)

# Predict button
if st.button("🔍 Predict Diabetes Risk"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("---")
    st.subheader("🩺 Prediction Result")

    if prediction[0] == 1:
        if lottie_warning:
            st_lottie(lottie_warning, height=150, key="warning")
        st.error("🔴 High Risk: The person is likely diabetic.")
    else:
        if lottie_success:
            st_lottie(lottie_success, height=150, key="success")
        st.success("🟢 Low Risk: The person is not likely diabetic.")

# Footer
st.markdown("---")
st.markdown("<center>❤️</center>", unsafe_allow_html=True)
