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

# --- Initialize / reinitialize data when num_samples changes ---
if "num_samples" not in st.session_state or st.session_state.num_samples != num_samples:
    st.session_state.num_samples = num_samples
    cols = ["Time"] + [f"Sample {i+1}" for i in range(num_samples)]
    st.session_state.df = pd.DataFrame(columns=cols)

# --- App UI ---
st.title("⏱️ Real-Time Time-to-Threshold for Multiple Samples")
st.subheader("Add new measurement:")

# --- Manual data entry ---
new_time = st.number_input(
    "Time (h):", value=0.0, step=0.1, format="%.2f"
)

# generate one input per sample
new_values = []
for i in range(num_samples):
    v = st.number_input(
        f"Sample {i+1} signal:",
        min_value=signal_min,
        max_value=signal_max,
        value=signal_min,
        step=(signal_max - signal_min) / 100,
        format="%.2f",
        key=f"input_sample_{i}"
    )
    new_values.append(v)

if st.button("Add Data Point"):
    df = st.session_state.df
    row = {"Time": new_time}
    for i, v in enumerate(new_values):
        row[f"Sample {i+1}"] = v
    df.loc[len(df)] = row
    st.session_state.df = df

# --- Editable data table ---
st.subheader("Collected Data (editable)")
edited = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor"
)
st.session_state.df = edited

# --- Threshold input ---
st.subheader("Threshold Setting")
default_thr = (signal_min + signal_max) / 2
thr_num = st.number_input(
    "Threshold (Signal) — enter value:",
    min_value=signal_min,
    max_value=signal_max,
    value=default_thr,
    step=(signal_max - signal_min) / 100,
    format="%.2f",
    key="thr_num"
)
thr_slider = st.slider(
    "Threshold (Signal) — or drag slider:",
    min_value=signal_min,
    max_value=signal_max,
    value=thr_num,
    step=(signal_max - signal_min) / 100,
    key="thr_slider"
)
threshold = thr_slider if thr_slider != thr_num else thr_num

# --- Prepare data ---
df = st.session_state.df.copy()
# drop rows where Time is missing or all samples are missing
mask = df["Time"].notna() & df[[f"Sample {i+1}" for i in range(num_samples)]].notna().any(axis=1)
df = df.loc[mask]

# --- Fit & plot for each sample when ≥5 points ---
if len(df) >= 5:
    time_arr = df["Time"].values
    sample_cols = [c for c in df.columns if c != "Time"]

    tabs = st.tabs(sample_cols)
    for idx, sample in enumerate(sample_cols):
        with tabs[idx]:
            st.subheader(sample)
            sig_arr = pd.to_numeric(df[sample], errors="coerce").values

            # decide linear fallback: no rise in first 12h
            if time_arr.max() >= 12 and (sig_arr[time_arr <= 12].max() - sig_arr[time_arr <= 12].min() <= 0):
                use_linear = True
            else:
                use_linear = False

            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(time_arr, sig_arr, 'ko', label="Data")

            if use_linear:
                # Linear fit
                m, b = np.polyfit(time_arr, sig_arr, 1)
                y_fit = m*time_arr + b
                # CI for linear
                resid = sig_arr - y_fit
                s_err = np.sqrt(np.sum(resid**2)/(len(time_arr)-2))
                tval = 2.262
                ci = tval * s_err * np.sqrt(
                    1/len(time_arr) + ((time_arr - time_arr.mean())**2)/np.sum((time_arr - time_arr.mean())**2)
                )
                ci_low = y_fit - ci
                ci_high = y_fit + ci

                ax.plot(time_arr, y_fit, 'b--', label="Linear Fit")
                ax.fill_between(time_arr, ci_low, ci_high, color='r', alpha=0.2)

                t_thresh = (threshold - b) / m if m != 0 else np.nan
                st.metric("⚠️ Linear fallback", f"Tt = {t_thresh:.2f} h")

            else:
                # 5PL with fixed asymptotes
                A, D = signal_min, signal_max
                def logistic_fixed(x, C, B, G):
                    return D - (D - A)/((1 + (x/C)**B)**G)

                p0 = [np.median(time_arr), 1.0, 1.0]
                popt, _ = curve_fit(
                    logistic_fixed, time_arr, sig_arr,
                    p0=p0,
                    bounds=([0,0,0],[np.inf,np.inf,np.inf]),
                    maxfev=10000
                )
                C_fit, B_fit, G_fit = popt

                t_plot = np.linspace(time_arr.min(), time_arr.max(), 200)
                y_plot = logistic_fixed(t_plot, *popt)

                resid = sig_arr - logistic_fixed(time_arr, *popt)
                s_err = np.std(resid)
                ci_low = y_plot - 1.96*s_err
                ci_high = y_plot + 1.96*s_err

                ax.plot(t_plot, y_plot, 'b-', label="5PL Fit")
                ax.fill_between(t_plot, ci_low, ci_high, color='r', alpha=0.2)

                def invert_5pl(y):
                    return C_fit * (((D - A)/(D - y))**(1/G_fit) - 1)**(1/B_fit)
                t_thresh = invert_5pl(threshold)
                st.metric("🔵 5PL fit", f"Tt = {t_thresh:.2f} h")

            # common styling
            ax.axhline(threshold, color='green', linestyle='--', linewidth=1)
            ax.axvline(t_thresh, color='orange', linestyle='--', linewidth=1)
            ax.set_xlim(time_arr.min(), time_arr.max())
            ax.set_ylim(signal_min, signal_max)
            ax.set_xlabel("Time (h)", fontweight='bold')
            ax.set_ylabel("Signal", fontweight='bold')
            ax.set_title(f"{sample} Fit & Threshold")
            st.pyplot(fig)

# --- Download data CSV ---
csv = st.session_state.df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download All Data as CSV",
    data=csv,
    file_name="multi_sample_data.csv",
    mime="text/csv"
)
