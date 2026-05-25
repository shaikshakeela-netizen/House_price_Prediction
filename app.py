import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests
import os
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Load Lottie animation
# -------------------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets10.lottiefiles.com/packages/lf20_j1adxtyb.json"
lottie_home = load_lottie_url(lottie_url)

# -------------------------------
# Load Model and Encoders
# -------------------------------
MODEL_FILE = "house_model.pkl"
cat_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']

if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found. Run 'train_model.py' first.")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

le_dict = {}
for col in cat_cols:
    with open(f"{col}_encoder.pkl", "rb") as f:
        le_dict[col] = pickle.load(f)

# -------------------------------
# Page Config + Dark Theme
# -------------------------------
st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
# ---------------------------------------------------
# 🌅 Gradient Background (Sunset Theme)
# ---------------------------------------------------
st.markdown("""
    <style>
    /* Full App Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(90deg, #FF9966, #FF5E62);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* 🌸 Global Italic Font */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-style: italic;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        color: white;
    }

    /* Headings */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }

    /* Input boxes and select boxes */
    .stNumberInput, .stSelectbox, .stSlider, .stTextInput {
        background-color: rgba(255,255,255,0.9) !important;
        border-radius: 12px;
        padding: 10px;
        color: black;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ffffff, #ffe0b2);
        color: #FF5E62;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #FFCCBC, #FFF3E0);
        color: #FF3D00;
    }

    /* Cards and boxes */
    .main-card {
        background-color: rgba(255,255,255,0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        color: #333;
    }

    /* Prediction box */
    .prediction-card {
        background: rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    </style>
""", unsafe_allow_html=True)


st.title("House Price Predictor")
st_lottie(lottie_home, height=250)

st.markdown("### Enter House Details:")

# -------------------------------
# Input Fields
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1200)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    stories = st.slider("Stories", 1, 3, 1)
    mainroad = st.selectbox("Main Road", ["yes","no"])
    guestroom = st.selectbox("Guest Room", ["yes","no"])

with col2:
    basement = st.selectbox("Basement", ["yes","no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes","no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes","no"])
    parking = st.slider("Parking", 0, 5, 1)
    prefarea = st.selectbox("Preferred Area", ["yes","no"])
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished","semi-furnished","unfurnished"])

# Encode categorical inputs
mainroad = le_dict['mainroad'].transform([mainroad])[0]
guestroom = le_dict['guestroom'].transform([guestroom])[0]
basement = le_dict['basement'].transform([basement])[0]
hotwaterheating = le_dict['hotwaterheating'].transform([hotwaterheating])[0]
airconditioning = le_dict['airconditioning'].transform([airconditioning])[0]
prefarea = le_dict['prefarea'].transform([prefarea])[0]
furnishingstatus = le_dict['furnishingstatus'].transform([furnishingstatus])[0]

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Price 💰"):
    input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                            hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${prediction:,.0f}")

    # -------------------------------
    # Interactive Input Feature Chart
    # -------------------------------
    feature_names = ["Area","Bedrooms","Bathrooms","Stories","Main Road","Guest Room","Basement",
                     "Hot Water Heating","Air Conditioning","Parking","Preferred Area","Furnishing Status"]
    feature_values = [area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                     hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]
    df_features = pd.DataFrame({"Feature": feature_names, "Value": feature_values})
    
    fig = px.bar(df_features, x="Feature", y="Value", title="Input Feature Values", text="Value",
                 color="Value", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Feature Importance Chart
# -------------------------------
st.markdown("### 🌟 Model Feature Importance")
if isinstance(model, RandomForestRegressor):
    importance = model.feature_importances_
    feature_names = ["area","bedrooms","bathrooms","stories","mainroad","guestroom","basement",
                     "hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"]
    df_importance = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    df_importance = df_importance.sort_values(by="Importance", ascending=False)
    
    fig2 = px.bar(df_importance, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Viridis", title="Feature Importance")
    st.plotly_chart(fig2, use_container_width=True)
