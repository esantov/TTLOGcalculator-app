import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
import os
import pickle

st.set_page_config(layout="wide")
st.title("⏱️ Real-Time Tt + logCFU/mL Calculator")

# --- Sidebar Setup ---
st.sidebar.title("Configuration")
num_samples = st.sidebar.number_input("Number of samples", min_value=1, max_value=10, value=2)
sample_names = []
for i in range(1, num_samples + 1):
    sample_names.append(st.sidebar.text_input(f"Sample {i} name", value=f"Sample {i}", key=f"name_{i}"))

# Sync sample names
if len(sample_names) == len(st.session_state.samples):
    for i, new_name in enumerate(sample_names):
        old_name = list(st.session_state.samples.keys())[i]
        if new_name != old_name:
            st.session_state.samples[new_name] = st.session_state.samples.pop(old_name)

# --- Session Save/Load UI ---
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save Session"):
    save_session(st.session_state.samples)
    st.sidebar.success("Session saved to file.")

if st.sidebar.button("📂 Load Session"):
    restored = load_session()
    if restored:
        st.session_state.samples = restored
        st.sidebar.success("Session loaded.")

# --- Per-sample Loop ---
summary_rows = []

for sample_name, state in st.session_state.samples.items():
    with st.expander(sample_name, expanded=True):
        # Config
        c1, c2, c3 = st.columns(3)
        smin = c1.number_input(f"{sample_name} Min", value=state["min"], format="%.2f", key=f"{sample_name}_min")
        smax = c2.number_input(f"{sample_name} Max", value=state["max"], format="%.2f", key=f"{sample_name}_max")
        thr = c3.number_input(f"{sample_name} Threshold", min_value=smin, max_value=smax,
                              value=state["threshold"], format="%.2f", key=f"{sample_name}_thr")
        state["min"], state["max"], state["threshold"] = smin, smax, thr

        # Calibration (with safe defaults)
        state.setdefault("use_cal", False)
        state.setdefault("a", -0.45)
        state.setdefault("b", 9.2)
        state.setdefault("cal_name", "")

        st.markdown("### 🧪 Calibration")
        cal1, cal2, cal3 = st.columns([1, 1, 2])
        state["use_cal"] = st.checkbox("Use calibration", value=state["use_cal"], key=f"{sample_name}_cal")

        state["a"] = cal1.number_input("a", value=state["a"], format="%.4f", key=f"{sample_name}_a")
        state["b"] = cal2.number_input("b", value=state["b"], format="%.4f", key=f"{sample_name}_b")
        state["cal_name"] = cal3.text_input("Calibration name", value=state["cal_name"], key=f"{sample_name}_calname")

        # Data input
        st.markdown("### ➕ Add Data")
        c1, c2, c3 = st.columns([1, 1, 1])
        new_time = c1.number_input(f"{sample_name} Time", value=0.0, format="%.2f", key=f"{sample_name}_time")
        new_signal = c2.number_input(f"{sample_name} Signal", min_value=smin, max_value=smax,
                                     value=smin, format="%.2f", key=f"{sample_name}_signal")
        if c3.button(f"Add", key=f"{sample_name}_add"):
            row = {"Time": new_time, "Signal": new_signal}
            state["df"] = pd.concat([state["df"], pd.DataFrame([row])], ignore_index=True)

        # Editable table
        st.markdown("### 📋 Data Table")
        df = st.data_editor(state["df"], hide_index=True, num_rows="dynamic", use_container_width=True,
                            key=f"editor_{sample_name}")
        state["df"] = df.reset_index(drop=True)

        # Fit
        clean = df.dropna(subset=["Time", "Signal"])
        if len(clean) < 5:
            st.warning("Need at least 5 points.")
            continue

        t_arr = clean["Time"].astype(float).values
        y_arr = clean["Signal"].astype(float).values
        use_linear = t_arr.max() >= 12 and (y_arr[t_arr <= 12].max() - y_arr[t_arr <= 12].min() <= 0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_arr, y_arr, "ko", label="Data")

        if use_linear:
            m, b = np.polyfit(t_arr, y_arr, 1)
            y_fit = m * t_arr + b
            resid = y_arr - y_fit
            s_err = np.sqrt(np.sum(resid**2) / (len(t_arr) - 2))
            tval = 2.262
            ci = tval * s_err * np.sqrt(1 / len(t_arr) + ((t_arr - t_arr.mean())**2) / np.sum((t_arr - t_arr.mean())**2))
            ax.plot(t_arr, y_fit, "b--")
            ax.fill_between(t_arr, y_fit - ci, y_fit + ci, color="r", alpha=0.2)
            t_thresh = (thr - b) / m if m != 0 else np.nan
            st.metric("⚠️ Linear fit", f"Tt = {t_thresh:.2f} h")
        else:
            A, D = smin, smax
            def five_pl(x, C, B, G):
                return D - (D - A) / ((1 + (x / C)**B)**G)
            popt, _ = curve_fit(five_pl, t_arr, y_arr,
                                p0=[np.median(t_arr), 1.0, 1.0],
                                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                                maxfev=10000)
            C_fit, B_fit, G_fit = popt
            t_plot = np.linspace(t_arr.min(), t_arr.max(), 200)
            y_plot = five_pl(t_plot, *popt)
            resid = y_arr - five_pl(t_arr, *popt)
            s_err = np.std(resid)
            ax.plot(t_plot, y_plot, "b-")
            ax.fill_between(t_plot, y_plot - 1.96 * s_err, y_plot + 1.96 * s_err, color="r", alpha=0.2)
            t_thresh = C_fit * (((D - A) / (D - thr))**(1 / G_fit) - 1)**(1 / B_fit)
            st.metric("🔵 5PL fit", f"Tt = {t_thresh:.2f} h")

        logcfu = np.nan
        if state["use_cal"] and not np.isnan(t_thresh):
            logcfu = state["a"] * t_thresh + state["b"]
            st.metric("🧪 logCFU/mL", f"{logcfu:.2f}")

        ax.axhline(thr, color="green", linestyle="--")
        ax.axvline(t_thresh, color="orange", linestyle="--")
        ax.set_title(f"{sample_name} Fit & Threshold")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Signal")
        ax.set_ylim(smin, smax)
        st.pyplot(fig)

        summary_rows.append({
            "Sample": sample_name,
            "Tt (h)": round(t_thresh, 2) if not np.isnan(t_thresh) else "",
            "logCFU/mL": round(logcfu, 2) if not np.isnan(logcfu) else "",
            "Calibration": state["cal_name"] if state["use_cal"] else ""
        })

# --- Export ZIP with summary + plot ---
summary_df = pd.DataFrame(summary_rows)
raw_df = pd.concat(
    [st.session_state.samples[name]["df"].assign(Sample=name) for name in st.session_state.samples],
    ignore_index=True
)

zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, 'w') as zf:
    zf.writestr("summary_logCFU.csv", summary_df.to_csv(index=False))
    zf.writestr("raw_data.csv", raw_df.to_csv(index=False))

    if not summary_df["logCFU/mL"].isnull().all():
        fig, ax = plt.subplots(figsize=(6, 4))
        valid = summary_df.dropna(subset=["logCFU/mL"])
        ax.bar(valid["Sample"], valid["logCFU/mL"], color="skyblue")
        ax.set_ylabel("logCFU/mL")
        ax.set_title("logCFU/mL per Sample")
        plt.tight_layout()
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png")
        img_buf.seek(0)
        zf.writestr("logCFU_plot.png", img_buf.read())

zip_buffer.seek(0)
st.download_button(
    label="📦 Download ZIP (Summary + logCFU Plot)",
    data=zip_buffer,
    file_name="logCFU_summary_export.zip",
    mime="application/zip"
)
