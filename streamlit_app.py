import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- Sidebar: configuration ---
st.sidebar.title("Configuration")

# Number of samples
num_samples = st.sidebar.number_input(
    "Number of samples", min_value=1, max_value=10, value=2, step=1
)

# Sensor signal range
st.sidebar.subheader("Sensor Signal Range")
signal_min = st.sidebar.number_input("Minimum signal", value=0.0, format="%.2f")
signal_max = st.sidebar.number_input("Maximum signal", value=100.0, format="%.2f")

# --- Initialize / reset DataFrame when num_samples changes ---
if "num_samples" not in st.session_state or st.session_state.num_samples != num_samples:
    st.session_state.num_samples = num_samples
    cols = ["Time"] + [f"Sample {i+1}" for i in range(num_samples)]
    st.session_state.df = pd.DataFrame(columns=cols)

# --- App title ---
st.title("⏱️ Real-Time Time-to-Threshold for Multiple Samples")

# --- Data entry: editable table ---
st.subheader("Enter or edit your data")
st.write(
    "You can add/remove rows or click into any cell to edit. "
    "First column is time (h), next columns are each sample's signal."
)
edited = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor"
)
st.session_state.df = edited

# --- Threshold input ---
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

# --- Only proceed when we have at least 5 rows with valid time & at least one signal ---
df = st.session_state.df.dropna(subset=["Time"] + [f"Sample {i+1}" for i in range(num_samples)])
if len(df) >= 5:
    # common time array
    time_arr = df["Time"].values

    # prepare tabs for each sample
    sample_cols = [col for col in df.columns if col != "Time"]
    tabs = st.tabs(sample_cols)

    # loop over each sample
    for idx, sample in enumerate(sample_cols):
        with tabs[idx]:
            st.subheader(f"{sample}")
            sig_arr = df[sample].values

            # decide linear fallback: no increase in first 12h
            if time_arr.max() >= 12:
                mask = time_arr <= 12
                increased = sig_arr[mask].max() - sig_arr[mask].min() > 0
                use_linear = not increased
            else:
                use_linear = False

            if use_linear:
                # linear fit
                m, b = np.polyfit(time_arr, sig_arr, 1)
                # time-to-threshold
                t_thresh = (threshold - b) / m if m != 0 else np.nan
                st.metric("⚠️ Linear fallback (no rise ≤12h)", f"Tt = {t_thresh:.2f} h")

                # confidence interval for linear fit
                # standard OLS CI (Gaussian)
                resid = sig_arr - (m * time_arr + b)
                s_err = np.sqrt(np.sum(resid**2) / (len(time_arr) - 2))
                tval = 2.262  # approx for 95% CI, df ~5
                ci_low = (m * time_arr + b) - tval * s_err * np.sqrt(1/len(time_arr) + ((time_arr - time_arr.mean())**2)/np.sum((time_arr - time_arr.mean())**2))
                ci_high = (m * time_arr + b) + tval * s_err * np.sqrt(1/len(time_arr) + ((time_arr - time_arr.mean())**2)/np.sum((time_arr - time_arr.mean())**2))

                fig, ax = plt.subplots(figsize=(6,6))
                ax.plot(time_arr, sig_arr, 'ko', label="Data")
                ax.plot(time_arr, m*time_arr + b, 'b--', label="Linear Fit")
                ax.fill_between(time_arr, ci_low, ci_high, color='r', alpha=0.2)
            else:
                # 5PL with fixed asymptotes
                A, D = signal_min, signal_max
                def logistic_fixed(x, C, B, G):
                    return D - (D - A) / ((1 + (x / C)**B)**G)

                p0 = [np.median(time_arr), 1.0, 1.0]
                popt, _ = curve_fit(
                    logistic_fixed,
                    time_arr,
                    sig_arr,
                    p0=p0,
                    bounds=([0,0,0],[np.inf,np.inf,np.inf]),
                    maxfev=10000
                )
                C_fit, B_fit, G_fit = popt

                # smooth curve and CI
                t_plot = np.linspace(time_arr.min(), time_arr.max(), 200)
                y_plot = logistic_fixed(t_plot, *popt)

                # approx CI by ±1.96*sd of residuals
                resid = sig_arr - logistic_fixed(time_arr, *popt)
                s_err = np.std(resid)
                ci_low = y_plot - 1.96*s_err
                ci_high = y_plot + 1.96*s_err

                # time-to-threshold
                def invert_5pl(y):
                    return C_fit * (((D - A)/(D - y))**(1/G_fit) - 1)**(1/B_fit)
                t_thresh = invert_5pl(threshold)
                st.metric("🔵 5PL fit", f"Tt = {t_thresh:.2f} h")

                fig, ax = plt.subplots(figsize=(6,6))
                ax.plot(time_arr, sig_arr, 'ko', label="Data")
                ax.plot(t_plot, y_plot, 'b-', label="5PL Fit")
                ax.fill_between(t_plot, ci_low, ci_high, color='r', alpha=0.2)

            # common plot elements
            ax.axhline(threshold, color='green', linestyle='--', linewidth=1)
            ax.axvline(t_thresh, color='orange', linestyle='--', linewidth=1)
            ax.set_xlim(time_arr.min(), time_arr.max())
            ax.set_ylim(signal_min, signal_max)
            ax.set_xlabel("Time (h)", fontweight='bold')
            ax.set_ylabel("Signal", fontweight='bold')
            ax.set_title(f"{sample} Fit & Threshold")
            st.pyplot(fig)

# --- Download raw data CSV ---
csv = st.session_state.df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download All Collected Data as CSV",
    data=csv,
    file_name="multi_sample_data.csv",
    mime="text/csv"
)
