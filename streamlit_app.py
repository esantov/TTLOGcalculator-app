# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import BytesIO
import zipfile
import os
import pickle
from datetime import datetime

# Setup
st.set_page_config(layout="wide")
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# Load/save helpers
def save_session(data, name):
    with open(os.path.join(SESSION_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f)

def load_session(name):
    with open(os.path.join(SESSION_DIR, name), "rb") as f:
        return pickle.load(f)

def list_sessions():
    return sorted([f for f in os.listdir(SESSION_DIR) if f.endswith(".pkl")], reverse=True)

# UI - Sample setup
st.sidebar.title("Configuration")
num_samples = st.sidebar.number_input("Number of samples", min_value=1, max_value=10, value=2)
sample_names = [st.sidebar.text_input(f"Sample {i+1} name", value=f"Sample {i+1}", key=f"name_{i}") for i in range(num_samples)]

# Initialize or load session
if "samples" not in st.session_state:
    if list_sessions():
        last = list_sessions()[0]
        st.session_state.samples = load_session(last)
        st.sidebar.success(f"Auto-loaded session: {last}")
    else:
        st.session_state.samples = {}
        for name in sample_names:
            st.session_state.samples[name] = {
                "df": pd.DataFrame(columns=["Time", "Signal"]),
                "min": 0.0, "max": 100.0, "threshold": 50.0,
                "use_cal": False, "a": -0.45, "b": 9.2, "cal_name": ""
            }

# Sync names
if len(sample_names) == len(st.session_state.samples):
    for i, new_name in enumerate(sample_names):
        old_name = list(st.session_state.samples.keys())[i]
        if new_name != old_name:
            st.session_state.samples[new_name] = st.session_state.samples.pop(old_name)

# Continue from previous cell
st.title("⏱️ Real-Time Tt + logCFU/mL Calculator")

summary_rows = []

for sample_name, state in st.session_state.samples.items():
    with st.expander(sample_name, expanded=True):
        # Setup fields
        smin = st.number_input(f"{sample_name} Min", value=state["min"], format="%.2f", key=f"{sample_name}_min")
        smax = st.number_input(f"{sample_name} Max", value=state["max"], format="%.2f", key=f"{sample_name}_max")
        thr = st.number_input(f"{sample_name} Threshold", min_value=smin, max_value=smax, value=state["threshold"],
                              format="%.2f", key=f"{sample_name}_thr")
        state.update({"min": smin, "max": smax, "threshold": thr})

        st.markdown("**Calibration**")
        state["use_cal"] = st.checkbox(f"Use calibration", value=state["use_cal"], key=f"{sample_name}_cal")
        state["a"] = st.number_input("a", value=state["a"], format="%.4f", key=f"{sample_name}_a")
        state["b"] = st.number_input("b", value=state["b"], format="%.4f", key=f"{sample_name}_b")
        state["cal_name"] = st.text_input("Calibration name", value=state["cal_name"], key=f"{sample_name}_calname")

        # Add data
        new_time = st.number_input(f"{sample_name} Time", value=0.0, format="%.2f", key=f"{sample_name}_time")
        new_signal = st.number_input(f"{sample_name} Signal", min_value=smin, max_value=smax,
                                     value=smin, format="%.2f", key=f"{sample_name}_signal")
        if st.button(f"Add point to {sample_name}", key=f"{sample_name}_add"):
            state["df"] = pd.concat([state["df"], pd.DataFrame([{"Time": new_time, "Signal": new_signal}])],
                                    ignore_index=True)

        st.data_editor(state["df"], hide_index=True, key=f"edit_{sample_name}")
        state["df"] = state["df"].reset_index(drop=True)

        df = state["df"].dropna(subset=["Time", "Signal"])
        if len(df) < 5:
            st.warning("Need at least 5 data points.")
            continue

        t, y = df["Time"].values, df["Signal"].values
        use_linear = t.max() >= 12 and (y[t <= 12].max() - y[t <= 12].min() <= 0)

        fig, ax = plt.subplots()
        ax.plot(t, y, 'ko')

        if use_linear:
            m, b = np.polyfit(t, y, 1)
            y_fit = m * t + b
            t_thresh = (thr - b) / m if m != 0 else np.nan
            ax.plot(t, y_fit, "b--")
            st.metric("⚠️ Linear fit", f"Tt = {t_thresh:.2f}")
        else:
            def five_pl(x, C, B, G):
                A, D = smin, smax
                return D - (D - A) / ((1 + (x / C) ** B) ** G)

            popt, _ = curve_fit(five_pl, t, y, p0=[np.median(t), 1, 1], bounds=([0, 0, 0], [np.inf]*3))
            t_fit = np.linspace(t.min(), t.max(), 200)
            y_fit = five_pl(t_fit, *popt)
            t_thresh = popt[0] * (((smax - smin) / (smax - thr)) ** (1 / popt[2]) - 1) ** (1 / popt[1])
            ax.plot(t_fit, y_fit, "b-")
            st.metric("🔵 5PL fit", f"Tt = {t_thresh:.2f}")

        ax.axhline(thr, color='green', linestyle='--')
        ax.axvline(t_thresh, color='orange', linestyle='--')
        st.pyplot(fig)

        logcfu = state["a"] * t_thresh + state["b"] if state["use_cal"] else np.nan
        if state["use_cal"]:
            st.metric("🧪 logCFU/mL", f"{logcfu:.2f}")
        summary_rows.append({
            "Sample": sample_name,
            "Tt (h)": round(t_thresh, 2),
            "logCFU/mL": round(logcfu, 2) if not np.isnan(logcfu) else "",
            "Calibration": state["cal_name"]
        })

# --- Export and save ---
summary_df = pd.DataFrame(summary_rows)
raw_df = pd.concat([s["df"].assign(Sample=n) for n, s in st.session_state.samples.items()])

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, index=False, sheet_name="Summary")
    raw_df.to_excel(writer, index=False, sheet_name="Raw Data")

zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, 'w') as zf:
    zf.writestr("logCFU_summary.xlsx", excel_buffer.getvalue())
    if not summary_df["logCFU/mL"].isnull().all():
        fig, ax = plt.subplots()
        valid = summary_df.dropna()
        ax.bar(valid["Sample"], valid["logCFU/mL"])
        ax.set_ylabel("logCFU/mL")
        ax.set_title("logCFU per Sample")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        zf.writestr("logCFU_plot.png", buf.getvalue())

st.download_button("📦 Download Results ZIP", zip_buffer.getvalue(), "logCFU_bundle.zip")

# Save/load sessions
if st.sidebar.button("💾 Save Session"):
    fname = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_session(st.session_state.samples, fname)
    st.sidebar.success(f"Saved session: {fname}")

sessions = list_sessions()
if sessions:
    sel = st.sidebar.selectbox("📂 Load Session", sessions)
    if st.sidebar.button("📥 Load Selected Session"):
        st.session_state.samples = load_session(sel)
        st.sidebar.success(f"Loaded: {sel}")

