import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ―― Sidebar: global config ――
st.sidebar.title("Global Configuration")
num_samples = st.sidebar.number_input("Number of samples", min_value=1, max_value=10, value=2, step=1)

# ―― Initialize per-sample state ――
if "samples" not in st.session_state or st.session_state.num_samples != num_samples:
    st.session_state.num_samples = num_samples
    st.session_state.samples = {}
    for i in range(1, num_samples + 1):
        name = f"Sample {i}"
        st.session_state.samples[name] = {
            "df": pd.DataFrame(columns=["Time", "Signal"]),
            "min": 0.0,
            "max": 100.0,
            "threshold": 50.0
        }

st.title("⏱️ Real‐Time Time-to-Threshold for Multiple Samples")

# ―― Loop over each sample in its own expander ――
for sample_name, state in st.session_state.samples.items():
    with st.expander(sample_name, expanded=True):
        # 1) Per-sample min/max/threshold
        c1, c2, c3 = st.columns(3)
        with c1:
            smin = st.number_input(f"{sample_name} Min", value=state["min"], format="%.2f", key=f"{sample_name}_min")
        with c2:
            smax = st.number_input(f"{sample_name} Max", value=state["max"], format="%.2f", key=f"{sample_name}_max")
        with c3:
            thr = st.number_input(
                f"{sample_name} Threshold",
                min_value=smin,
                max_value=smax,
                value=state["threshold"],
                format="%.2f",
                key=f"{sample_name}_thr"
            )
        state["min"], state["max"], state["threshold"] = smin, smax, thr

        # 2) Editable table, hide the index to prevent new index-column creation
        st.markdown("**Time‐Signal Data** (add/edit rows)")
        df = st.data_editor(
    state["df"],
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    key=f"editor_{sample_name}"
)

        state["df"] = df

        # 3) Fit & plot once ≥5 points
        clean = df.dropna(subset=["Time", "Signal"])
        if len(clean) < 5:
            st.warning(f"Need ≥5 points to fit ({len(clean)} present)")
            continue

        t_arr = clean["Time"].astype(float).values
        y_arr = clean["Signal"].astype(float).values

        # decide linear fallback
        if t_arr.max() >= 12 and (y_arr[t_arr <= 12].max() - y_arr[t_arr <= 12].min() <= 0):
            use_linear = True
        else:
            use_linear = False

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_arr, y_arr, "ko", label="Data")

        if use_linear:
            # linear fit + CI
            m, b = np.polyfit(t_arr, y_arr, 1)
            y_fit = m * t_arr + b
            resid = y_arr - y_fit
            s_err = np.sqrt(np.sum(resid**2) / (len(t_arr) - 2))
            tval = 2.262
            ci = tval * s_err * np.sqrt(
                1 / len(t_arr) + ((t_arr - t_arr.mean()) ** 2) / np.sum((t_arr - t_arr.mean()) ** 2)
            )
            ax.plot(t_arr, y_fit, "b--", label="Linear Fit")
            ax.fill_between(t_arr, y_fit - ci, y_fit + ci, color="r", alpha=0.2)

            t_thresh = (thr - b) / m if m != 0 else np.nan
            st.metric("⚠️ Linear fallback", f"Tt = {t_thresh:.2f} h")
        else:
            # 5PL with fixed asymptotes
            A, D = state["min"], state["max"]
            def five_pl(x, C, B, G):
                return D - (D - A) / ((1 + (x / C) ** B) ** G)

            popt, _ = curve_fit(
                five_pl, t_arr, y_arr,
                p0=[np.median(t_arr), 1.0, 1.0],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                maxfev=10000
            )
            C_fit, B_fit, G_fit = popt
            t_plot = np.linspace(t_arr.min(), t_arr.max(), 200)
            y_plot = five_pl(t_plot, *popt)

            resid = y_arr - five_pl(t_arr, *popt)
            s_err = np.std(resid)
            ax.plot(t_plot, y_plot, "b-", label="5PL Fit")
            ax.fill_between(t_plot, y_plot - 1.96 * s_err, y_plot + 1.96 * s_err, color="r", alpha=0.2)

            t_thresh = C_fit * (((D - A) / (D - thr)) ** (1 / G_fit) - 1) ** (1 / B_fit)
            st.metric("🔵 5PL fit", f"Tt = {t_thresh:.2f} h")

        # common styling
        ax.axhline(thr, color="green", linestyle="--", linewidth=1)
        ax.axvline(t_thresh, color="orange", linestyle="--", linewidth=1)
        ax.set_xlabel("Time (h)", fontweight="bold")
        ax.set_ylabel("Signal", fontweight="bold")
        ax.set_title(f"{sample_name} Fit & Threshold")
        ax.set_ylim(state["min"], state["max"])
        st.pyplot(fig)

# ―― Download all data ――
all_df = pd.concat(
    [st.session_state.samples[name]["df"].assign(Sample=name)
     for name in st.session_state.samples],
    ignore_index=True
)
csv = all_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download All Samples Data as CSV",
    data=csv,
    file_name="all_samples_data.csv",
    mime="text/csv"
)
