import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============ Load Model & Encoders ============
model = joblib.load('crop_yield_model.pkl')
le_state = joblib.load('le_state.pkl')
le_district = joblib.load('le_district.pkl')
le_crop = joblib.load('le_crop.pkl')
le_season = joblib.load('le_season.pkl')

# ============ Page Config ============
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")

st.title("🌾 Crop Yield Prediction App")
st.write("Enter the details below to predict crop yield")

# ============ User Inputs ============
col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", le_state.classes_)
    district = st.selectbox("District", le_district.classes_)
    crop = st.selectbox("Crop", le_crop.classes_)
    season = st.selectbox("Season", le_season.classes_)

with col2:
    year = st.number_input("Year", min_value=1997, max_value=2030, value=2023)
    area = st.number_input("Area (Hectares)", min_value=0.0, value=1000.0)
    production = st.number_input("Production (Tonnes)", min_value=0.0, value=1000.0)

# ============ Prediction ============
if st.button("🔍 Predict Yield"):
    try:
        # Encode categorical inputs
        state_enc = le_state.transform([state])[0]
        district_enc = le_district.transform([district])[0]
        crop_enc = le_crop.transform([crop])[0]
        season_enc = le_season.transform([season])[0]

        # Create input array (SAME ORDER as training: State, District, Crop, Year, Season, Area, Production)
        input_data = np.array([[state_enc, district_enc, crop_enc, year, season_enc, area, production]])

        # Predict
        prediction = model.predict(input_data)

        st.success(f"### 🌾 Predicted Yield: {prediction[0]:.2f} tons/hectare")

    except Exception as e:
        st.error(f"Error: {e}")

# ============ Footer ============
st.markdown("---")
st.caption("Built with Random Forest Regressor | Crop Yield Prediction Project")