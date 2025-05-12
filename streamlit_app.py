import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# — Sidebar: global config —
st.sidebar.title("Global Configuration")
num_samples = st.sidebar.number_input("Number of samples", min_value=1, max_value=10, value=2, step=1)

# Initialize per-sample state
if "samples" not in st.session_state or st.session_state.num_samples != num_samples:
    st.session_state.num_samples = num_samples
    # For each sample, store a DataFrame and its params
    st.session_state.samples = {}
    for i in range(1, num_samples+1):
        name = f"Sample {i}"
        st.session_state.samples[name] = {
            "df": pd.DataFrame(columns=["Time", "Signal"]),
            "min": 0.0,
            "max": 100.0,
            "threshold": 50.0
        }

st.title("⏱️ Real‐Time Time-to-Threshold for Multiple Samples")

# Loop over samples
for sample_name, state in st.session_state.samples.items():
    with st.expander(sample_name, expanded=True):
        # 1) Per-sample parameter inputs
        st.markdown("**Signal Range & Threshold**")
        col1, col2, col3 = st.columns(3)
        with col1:
            smin = st.number_input(f"{sample_name} Min signal",
                                   value=state["min"], format="%.2f",
                                   key=f"{sample_name}_min")
        with col2:
            smax = st.number_input(f"{sample_name} Max signal",
                                   value=state["max"], format="%.2f",
                                   key=f"{sample_name}_max")
        with col3:
            thr = st.number_input(f"{sample_name} Threshold",
                                  min_value=smin, max_value=smax,
                                  value=state["threshold"], format="%.2f",
                                  key=f"{sample_name}_thr")
        # Save back
        state["min"], state["max"], state["threshold"] = smin, smax, thr

        # 2) Data entry / editing
        st.markdown("**Time-Signal Data** (add/edit rows)")
        df = st.data_editor(
            state["df"],
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{sample_name}"
        )
        state["df"] = df

        # 3) Fit and plot if we have ≥5 valid points
        clean = df.dropna(subset=["Time", "Signal"])
        if len(clean) < 5:
            st.warning(f"Need at least 5 points to fit (have {len(clean)})")
            continue

        t_arr = clean["Time"].astype(float).values
        y_arr = clean["Signal"].astype(float).values

        # Decide linear fallback: if no increase in first 12h
        if t_arr.max() >= 12 and (y_arr[t_arr <= 12].max() - y_arr[t_arr <= 12].min() <= 0):
            use_linear = True
        else:
            use_linear = False

        # Prepare figure
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(t_arr, y_arr, 'ko', label="Data")

        if use_linear:
            # Linear regression
            m, b = np.polyfit(t_arr, y_arr, 1)
            y_fit = m*t_arr + b
            # CI for linear
            resid = y_arr - y_fit
            s_err = np.sqrt(np.sum(resid**2)/(len(t_arr)-2))
            tval = 2.262  # approx for df~5
            ci = tval * s_err * np.sqrt(
                1/len(t_arr) + ((t_arr - t_arr.mean())**2)/np.sum((t_arr - t_arr.mean())**2)
            )
            ci_low = y_fit - ci
            ci_high = y_fit + ci

            ax.plot(t_arr, y_fit, 'b--', label="Linear Fit")
            ax.fill_between(t_arr, ci_low, ci_high, color='r', alpha=0.2)

            # Predict Tt
            t_thresh = (thr - b)/m if m!=0 else np.nan
            st.metric("⚠️ Linear fallback", f"Time-to-Threshold = {t_thresh:.2f} h")

        else:
            # 5PL with fixed asymptotes = smin, smax
            A, D = smin, smax
            def five_pl(x, C, B, G):
                return D - (D - A)/((1 + (x/C)**B)**G)

            p0 = [np.median(t_arr), 1.0, 1.0]
            popt, _ = curve_fit(
                five_pl, t_arr, y_arr, p0=p0,
                bounds=([0,0,0],[np.inf,np.inf,np.inf]),
                maxfev=10000
            )
            C_fit, B_fit, G_fit = popt
            t_plot = np.linspace(t_arr.min(), t_arr.max(), 200)
            y_plot = five_pl(t_plot, *popt)

            # CI by ±1.96*sd(resid)
            resid = y_arr - five_pl(t_arr, *popt)
            s_err = np.std(resid)
            ci_low = y_plot - 1.96*s_err
            ci_high = y_plot + 1.96*s_err

            ax.plot(t_plot, y_plot, 'b-', label="5PL Fit")
            ax.fill_between(t_plot, ci_low, ci_high, color='r', alpha=0.2)

            # Predict Tt via closed-form inversion
            def invert_5pl(y):
                return C_fit*(((D - A)/(D - y))**(1/G_fit) - 1)**(1/B_fit)
            t_thresh = invert_5pl(thr)
            st.metric("🔵 5PL fit", f"Time-to-Threshold = {t_thresh:.2f} h")

        # final styling
        ax.axhline(thr, color='green', linestyle='--', linewidth=1)
        ax.axvline(t_thresh, color='orange', linestyle='--', linewidth=1)
        ax.set_xlim(t_arr.min(), t_arr.max())
        ax.set_ylim(smin, smax)
        ax.set_xlabel("Time (h)", fontweight='bold')
        ax.set_ylabel("Signal", fontweight='bold')
        ax.set_title(f"{sample_name} Fit & Threshold")
        st.pyplot(fig)

# — Download all raw data —
all_df = pd.concat(
    [st.session_state.samples[name]["df"].assign(Sample=name)
     for name in st.session_state.samples],
    ignore_index=True
)
csv = all_df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download All Samples Data as CSV",
    data=csv,
    file_name="all_samples_data.csv",
    mime="text/csv"
)
