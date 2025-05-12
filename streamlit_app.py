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

# Time‐grid configuration
st.sidebar.subheader("Time Grid (h)")
time_start = st.sidebar.number_input("Start", value=0.0, format="%.2f")
time_end   = st.sidebar.number_input("End",   value=10.0, format="%.2f")
time_step  = st.sidebar.number_input("Step",  value=1.0, format="%.2f")

# --- Initialize / reinitialize session_state.df when config changes ---
config = (num_samples, time_start, time_end, time_step)
if ("config" not in st.session_state) or (st.session_state.config != config):
    st.session_state.config = config
    # build time points
    time_points = np.arange(time_start, time_end + time_step/2, time_step)
    # build empty dataframe
    cols = ["Time"] + [f"Sample {i+1}" for i in range(num_samples)]
    df = pd.DataFrame(index=range(len(time_points)), columns=cols)
    df["Time"] = time_points
    st.session_state.df = df

# --- App title ---
st.title("⏱️ Real-Time Time-to-Threshold for Multiple Samples")

# --- Editable data table with fixed Time column ---
st.subheader("Enter or edit your data")
st.write("First column (Time) is auto-filled; edit only the sample signals.")
edited = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    disabled=["Time"],           # prevent editing of Time column
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

# --- Proceed if at least 5 time points and at least one non-empty sample ---
df = st.session_state.df.copy()
# drop rows where all sample columns are NaN
sample_cols = df.columns.drop("Time")
has_data = df[sample_cols].notna().any(axis=1)
df = df.loc[has_data]

if len(df) >= 5:
    time_arr = df["Time"].values

    # Create tabs per sample
    tabs = st.tabs(sample_cols)
    for idx, sample in enumerate(sample_cols):
        with tabs[idx]:
            st.subheader(sample)
            sig = df[sample].values

            # decide linear fallback: no increase in first 12h
            if time_arr.max() >= 12:
                mask = time_arr <= 12
                increased = (sig[mask].max() - sig[mask].min()) > 0
                use_linear = not increased
            else:
                use_linear = False

            if use_linear:
                # --- Linear fit ---
                m, b = np.polyfit(time_arr, sig, 1)
                t_thresh = (threshold - b) / m if m != 0 else np.nan
                st.metric("⚠️ Linear fallback (no rise ≤12h)",
                          f"Time-to-Threshold = {t_thresh:.2f} h")
                # CI for linear
                resid = sig - (m*time_arr + b)
                s_err = np.sqrt(np.sum(resid**2)/(len(time_arr)-2))
                tval = 2.262  # approx for 95% CI, df ~5
                ci_low = (m*time_arr + b) - tval*s_err*np.sqrt(
                    1/len(time_arr) + ((time_arr - time_arr.mean())**2)/np.sum((time_arr - time_arr.mean())**2))
                ci_high = (m*time_arr + b) + tval*s_err*np.sqrt(
                    1/len(time_arr) + ((time_arr - time_arr.mean())**2)/np.sum((time_arr - time_arr.mean())**2))
                fig, ax = plt.subplots(figsize=(6,6))
                ax.plot(time_arr, sig, 'ko', label="Data")
                ax.plot(time_arr, m*time_arr + b, 'b--', label="Linear Fit")
                ax.fill_between(time_arr, ci_low, ci_high, color='r', alpha=0.2)
            else:
                # --- 5PL with fixed asymptotes ---
                A, D = signal_min, signal_max
                def logistic_fixed(x, C, B, G):
                    return D - (D - A) / ((1 + (x/C)**B)**G)
                p0 = [np.median(time_arr), 1.0, 1.0]
                popt, _ = curve_fit(
                    logistic_fixed,
                    time_arr,
                    sig,
                    p0=p0,
                    bounds=([0,0,0],[np.inf,np.inf,np.inf]),
                    maxfev=10000
                )
                C_fit, B_fit, G_fit = popt
                t_plot = np.linspace(time_arr.min(), time_arr.max(), 200)
                y_plot = logistic_fixed(t_plot, *popt)
                # CI for 5PL (±1.96*sd of residuals)
                resid = sig - logistic_fixed(time_arr, *popt)
                s_err = np.std(resid)
                ci_low = y_plot - 1.96*s_err
                ci_high = y_plot + 1.96*s_err
                def invert_5pl(y):
                    return C_fit * (((D - A)/(D - y))**(1/G_fit) - 1)**(1/B_fit)
                t_thresh = invert_5pl(threshold)
                st.metric("🔵 5PL fit", f"Time-to-Threshold = {t_thresh:.2f} h")
                fig, ax = plt.subplots(figsize=(6,6))
                ax.plot(time_arr, sig, 'ko', label="Data")
                ax.plot(t_plot, y_plot, 'b-', label="5PL Fit")
                ax.fill_between(t_plot, ci_low, ci_high, color='r', alpha=0.2)

            # common styling
            ax.axhline(threshold, color='green', linestyle='--', linewidth=1)
            ax.axvline(t_thresh, color='orange', linestyle='--', linewidth=1)
            ax.set_xlim(time_arr.min(), time_arr.max())
            ax.set_ylim(signal_min, signal_max)
            ax.set_xlabel("Time (h)", fontweight='bold')
            ax.set_ylabel("Signal", fontweight='bold')
            ax.set_title(f"{sample} Fit & Threshold")
            st.pyplot(fig)

# --- Download raw data ---
csv = st.session_state.df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download All Collected Data as CSV",
    data=csv,
    file_name="multi_sample_data.csv",
    mime="text/csv",
)
