import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import fit_capacity_fade, predict_cycles_to_eol
from utils import clean_data

st.title("🔋 Battery Life Predictor (SOH Forecasting Tool)")
st.write("Upload your battery cycling data to predict capacity fade and estimated lifetime.")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ---------- Data Cleaning ----------
    df_clean = clean_data(df)

    st.subheader("Cleaned Data")
    st.dataframe(df_clean.head())

    cycles = df_clean["cycle"].values
    capacity = df_clean["capacity"].values

    # ---------- Model Fitting ----------
    st.subheader("Model Fitting")
    params = fit_capacity_fade(cycles, capacity)

    st.write("Model Parameters:")
    st.json({
        "A1": params[0],
        "k1": params[1],
        "A2": params[2],
        "k2": params[3],
        "Baseline Capacity": params[4]
    })

    # ---------- Plotting ----------
    st.subheader("Capacity Fade Curve")

    fig, ax = plt.subplots()
    ax.scatter(cycles, capacity, label="Measured", s=15)

    fitted = params[0] * np.exp(-params[1]*cycles) + \
             params[2] * np.exp(-params[3]*cycles) + params[4]

    ax.plot(cycles, fitted, label="Fitted Model", linewidth=2)
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Capacity (mAh)")
    ax.legend()
    st.pyplot(fig)

    # ---------- Lifetime Prediction ----------
    st.subheader("Lifetime Prediction (to 80% SOH)")
    eol_cycle = predict_cycles_to_eol(params, soh_threshold=0.8)


    st.success(f"Estimated life until 80% SOH: **{int(eol_cycle)} cycles**")

    # ---------- User-adjusted factors ----------
    st.subheader("Stress Factor Adjustment")
    fast_charge = st.slider("Fast Charge Rate (1C–5C)", 1.0, 5.0, 1.0)
    temp = st.slider("Temperature (°C)", 20, 60, 25)

    adjusted_life = eol_cycle / (fast_charge * (1 + 0.02*(temp-25)))
    st.info(f"Adjusted life prediction: **{int(adjusted_life)} cycles**")

    # ---------- Automatic Engineering Summary ----------
    st.subheader("Engineering Summary")

    st.write(
        f"""
        **Based on the uploaded data and fitted degradation model:**
        - Baseline life estimate: **{int(eol_cycle)} cycles**
        - Under fast-charging at {fast_charge}C and {temp}°C:  
          → Adjusted life: **{int(adjusted_life)} cycles**
        - Higher C-rate and elevated temperature accelerate SEI thickening and lithium plating,  
          leading to faster capacity fade.
        
        **This tool demonstrates SOH forecasting using real data and machine-learning inspired modeling.**
        """
    )


