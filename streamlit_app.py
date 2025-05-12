import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- Initialize session state ---
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["Time", "Signal"])

st.title("⏱️ Real-Time Time-to-Threshold Calculator")

# --- Sensor signal range controls ---
st.sidebar.subheader("Sensor Signal Range")
signal_min = st.sidebar.number_input(
    "Minimum signal", value=0.0, format="%.2f"
)
signal_max = st.sidebar.number_input(
    "Maximum signal", value=100.0, format="%.2f"
)

# --- Data entry ---
st.subheader("Add New Measurement")
c1, c2 = st.columns(2)
with c1:
    new_time = st.number_input("Time (h):", value=0.0, step=0.1, format="%.2f")
with c2:
    new_signal = st.number_input(
        "Signal:",
        min_value=signal_min,
        max_value=signal_max,
        value=np.clip(0.0, signal_min, signal_max),
        step=0.1,
        format="%.2f",
    )

if st.button("Add Data Point"):
    df = st.session_state.df
    df.loc[len(df)] = [new_time, new_signal]
    st.session_state.df = df

# --- Editable data table ---
st.subheader("Collected Data (editable)")
edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor",
)
st.session_state.df = edited_df

# --- Only proceed when ≥5 points collected ---
if len(st.session_state.df) >= 5:
    time_arr = st.session_state.df["Time"].values
    sig_arr = st.session_state.df["Signal"].values

    # Decide model: linear if no increase in the first 12 h
    if time_arr.max() >= 12:
        # restrict to times ≤12h
        mask = time_arr <= 12
        increased = sig_arr[mask].max() - sig_arr[mask].min() > 0
        use_linear = not increased
    else:
        use_linear = False

    # Dual threshold input
    st.subheader("Threshold Setting")
    default_thr = float((signal_min + signal_max) / 2)
    thr_num = st.number_input(
        "Threshold (Signal) — enter value:",
        min_value=signal_min,
        max_value=signal_max,
        value=default_thr,
        step=(signal_max - signal_min) / 100,
        format="%.2f",
    )
    thr_slider = st.slider(
        "Threshold (Signal) — or drag slider:",
        min_value=signal_min,
        max_value=signal_max,
        value=thr_num,
        step=(signal_max - signal_min) / 100,
    )
    if thr_slider != thr_num:
        thr_num = thr_slider
    threshold = thr_num

    st.subheader("Fit & Prediction")
    if use_linear:
        # --- Linear fallback ---
        m, b = np.polyfit(time_arr, sig_arr, 1)
        t_thresh = (threshold - b) / m if m != 0 else np.nan

        st.metric("⚠️ No increase detected → Linear fit", f"Time-to-Threshold = {t_thresh:.2f} h")

        # Plot linear
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(time_arr, sig_arr, "ko", label="Data")
        ax.plot(time_arr, m * time_arr + b, "b--", label="Linear Fit")
        ax.axhline(threshold, color="green", linestyle="--", linewidth=1, label="Threshold")
        ax.axvline(t_thresh, color="orange", linestyle="--", linewidth=1, label=f"Tt = {t_thresh:.2f} h")
    else:
        # --- 5PL growth fit with fixed asymptotes ---
        A, D = signal_min, signal_max

        def logistic_fixed(x, C, B, G):
            return D - (D - A) / ((1 + (x / C) ** B) ** G)

        # initial guesses for C, B, G
        p0 = [np.median(time_arr), 1.0, 1.0]
        popt, _ = curve_fit(
            logistic_fixed,
            time_arr,
            sig_arr,
            p0=p0,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        C_fit, B_fit, G_fit = popt

        # build smooth curve
        t_plot = np.linspace(time_arr.min(), time_arr.max(), 200)
        y_plot = logistic_fixed(t_plot, *popt)

        # invert for threshold
        def invert_5pl(y):
            return C_fit * (((D - A) / (D - y)) ** (1 / G_fit) - 1) ** (1 / B_fit)

        t_thresh = invert_5pl(threshold)
        st.metric("🔵 5PL fit", f"Time-to-Threshold = {t_thresh:.2f} h")

        # Plot 5PL
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(time_arr, sig_arr, "ko", label="Data")
        ax.plot(t_plot, y_plot, "b-", label="5PL Fit")
        ax.axhline(threshold, color="green", linestyle="--", linewidth=1, label="Threshold")
        ax.axvline(t_thresh, color="orange", linestyle="--", linewidth=1, label=f"Tt = {t_thresh:.2f} h")

    # finalize plot styling
    ax.set_xlim(time_arr.min(), time_arr.max())
    ax.set_ylim(signal_min, signal_max)
    ax.set_xlabel("Time (h)", fontweight="bold")
    ax.set_ylabel("Signal", fontweight="bold")
    ax.set_title("Model Fit & Time-to-Threshold")
    ax.legend()
    st.pyplot(fig)

# --- Download raw data as CSV ---
csv = st.session_state.df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Collected Data as CSV",
    data=csv,
    file_name="real_time_data.csv",
    mime="text/csv",
)
