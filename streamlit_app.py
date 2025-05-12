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
col1, col2 = st.columns(2)
with col1:
    new_time = st.number_input(
        "Time (h):", value=0.0, step=0.1, format="%.2f"
    )
with col2:
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

# --- Fit & plot when enough points are present ---
if len(st.session_state.df) >= 5:
    time_arr = st.session_state.df["Time"].values
    sig_arr = st.session_state.df["Signal"].values

    # 5PL growth model (increasing S-shape)
    def logistic_growth(x, A, D, C, B, G):
        return D - (D - A) / ((1 + (x / C) ** B) ** G)

    # Initial parameter guesses
    p0 = [
        np.min(sig_arr),     # A: lower asymptote
        np.max(sig_arr),     # D: upper asymptote
        np.median(time_arr), # C: inflection
        1.0,                  # B: slope
        1.0,                  # G: asymmetry
    ]

    st.subheader("5PL Fit & Prediction")
    try:
        popt, pcov = curve_fit(
            logistic_growth,
            time_arr,
            sig_arr,
            p0=p0,
            bounds=(
                [signal_min, signal_min, 0, 0, 0],
                [signal_max, signal_max, np.inf, np.inf, np.inf],
            ),
            maxfev=10000,
        )
        A, D, C, B, G = popt

        # Build smooth curve for plotting
        t_plot = np.linspace(time_arr.min(), time_arr.max(), 200)
        y_plot = logistic_growth(t_plot, *popt)

        # --- Dual threshold inputs ---
        default_thr = float((A + D) / 2)
        threshold_num = st.number_input(
            "Threshold (Signal) — enter a value:",
            min_value=float(signal_min),
            max_value=float(signal_max),
            value=default_thr,
            step=(signal_max - signal_min) / 100,
            format="%.2f",
        )
        threshold_slider = st.slider(
            "Threshold (Signal) — or drag the slider:",
            min_value=float(signal_min),
            max_value=float(signal_max),
            value=threshold_num,
            step=(signal_max - signal_min) / 100,
        )
        # Sync slider → number input
        if threshold_slider != threshold_num:
            threshold_num = threshold_slider
        threshold = threshold_num

        # Invert 5PL to get time-to-threshold
        def invert_5pl(y):
            return C * (((D - A) / (D - y)) ** (1 / G) - 1) ** (1 / B)

        t_thresh = invert_5pl(threshold)
        st.metric("⏱️ Predicted Time-to-Threshold (h)", f"{t_thresh:.2f}")

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(time_arr, sig_arr, "ko", label="Data")
        ax.plot(t_plot, y_plot, "b-", label="5PL Fit")
        ax.axhline(
            threshold, color="green", linestyle="--", linewidth=1, label="Threshold"
        )
        ax.axvline(
            t_thresh,
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"Tt = {t_thresh:.2f} h",
        )
        ax.set_xlim(time_arr.min(), time_arr.max())
        ax.set_ylim(signal_min, signal_max)
        ax.set_xlabel("Time (h)", fontweight="bold")
        ax.set_ylabel("Signal", fontweight="bold")
        ax.set_title("5PL Fit & Time-to-Threshold")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ 5PL fit failed: {e}")

# --- Download raw data as CSV ---
csv = st.session_state.df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Collected Data as CSV",
    data=csv,
    file_name="real_time_data.csv",
    mime="text/csv",
)
