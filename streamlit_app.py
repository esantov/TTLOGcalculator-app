import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- Initialize session state ---
if "df" not in st.session_state:
    # Store manual time/signal entries
    st.session_state.df = pd.DataFrame(columns=["Time", "Signal"])

st.title("⏱️ Real-Time Time-to-Threshold Calculator")
st.write("Enter each new (time, signal) measurement and see the 5PL fit & predicted time-to-threshold update in real time.")

# --- Data entry ---
col1, col2 = st.columns(2)
with col1:
    new_time = st.number_input("Time (h):", value=0.0, step=0.1, format="%.2f")
with col2:
    new_signal = st.number_input("Signal:", value=0.0, step=0.1, format="%.2f")

if st.button("Add Data Point"):
    df = st.session_state.df
    # Add new row without using deprecated .append()
    df.loc[len(df)] = [new_time, new_signal]
    st.session_state.df = df

# --- Show collected data ---
st.subheader("Collected Data")
st.dataframe(st.session_state.df)

# --- Only attempt fit if we have enough points ---
if len(st.session_state.df) >= 5:
    time_arr = st.session_state.df["Time"].values
    sig_arr  = st.session_state.df["Signal"].values

    # Define increasing 5PL (sigmoidal growth)
    def logistic_growth(x, A, D, C, B, G):
        return D - (D - A) / ((1 + (x / C) ** B) ** G)

    # Initial parameter guesses
    p0 = [
        np.min(sig_arr),        # A: lower asymptote
        np.max(sig_arr),        # D: upper asymptote
        np.median(time_arr),    # C: inflection
        1.0,                     # B: slope
        1.0                      # G: asymmetry
    ]

    st.subheader("5PL Fit & Prediction")
    try:
        # Fit the 5PL model
        popt, pcov = curve_fit(logistic_growth, time_arr, sig_arr, p0=p0, maxfev=10000)
        A, D, C, B, G = popt

        # Build a smooth curve for plotting
        t_plot = np.linspace(time_arr.min(), time_arr.max(), 200)
        y_plot = logistic_growth(t_plot, *popt)

        # Threshold input
        threshold = st.number_input("Threshold (Signal):", value=(A + D) / 2, step=0.1, format="%.2f")

        # Invert 5PL to solve for time-to-threshold
        def invert_5pl(y):
            return C * (((D - A) / (D - y)) ** (1 / G) - 1) ** (1 / B)

        t_thresh = invert_5pl(threshold)
        st.metric("⏱️ Predicted Time-to-Threshold (h)", f"{t_thresh:.2f}")

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 6))
        # Raw data
        ax.plot(time_arr, sig_arr, 'ko', label="Data")
        # 5PL fit curve
        ax.plot(t_plot, y_plot, 'b-', label="5PL Fit")
        # Threshold line
        ax.axhline(threshold, color='green', linestyle='--', linewidth=1, label="Threshold")
        # Predicted Tt vertical
        ax.axvline(t_thresh, color='orange', linestyle='--', linewidth=1, label=f"Tt = {t_thresh:.2f} h")
        ax.set_xlabel("Time (h)", fontweight='bold')
        ax.set_ylabel("Signal", fontweight='bold')
        ax.set_title("5PL Fit & Time-to-Threshold")
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ 5PL fit failed: {e}")

# --- Download raw data ---
csv = st.session_state.df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download Collected Data as CSV",
    data=csv,
    file_name="real_time_data.csv",
    mime="text/csv"
)
