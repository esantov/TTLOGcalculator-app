import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---- Initialize session state for data storage ----
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Time', 'Signal'])

st.title('⏱️ Real-Time Time-to-Threshold Calculator')
st.markdown('Enter time and sensor signal data points as they arrive to estimate the time to reach a given threshold using a 5PL fit.')

# ---- User inputs for new data point ----
col1, col2 = st.columns(2)
with col1:
    new_time = st.number_input('Time', value=0.0, step=0.01, format="%.2f", key='input_time')
with col2:
    new_signal = st.number_input('Signal', value=0.0, step=0.01, format="%.2f", key='input_signal')

if st.button('Add Data Point'):
    # Append new row to DataFrame
    st.session_state.df = st.session_state.df.append({'Time': new_time, 'Signal': new_signal}, ignore_index=True)

# ---- Display current data ----
st.subheader('Current Data')
st.dataframe(st.session_state.df)

# ---- Threshold setting ----nthreshold = st.number_input('Threshold', value=1.0, step=0.01, format="%.2f")

# ---- Define 5PL model ----
def logistic_5pl(x, A, D, C, B, G):
    return D - (D - A) / ((1 + (x / C)**B)**G)

def inverse_5pl(y, A, D, C, B, G):
    # Solve for x in logistic_5pl(x) = y
    ratio = (D - A) / (D - y)
    base = ratio**(1/G) - 1
    return C * (base)**(1/B)

# ---- Perform fit and plot ----
if len(st.session_state.df) >= 5:
    time_vals = st.session_state.df['Time'].values
    signal_vals = st.session_state.df['Signal'].values

    # Initial parameter guesses
    p0 = [min(signal_vals), max(signal_vals), np.median(time_vals), 1.0, 1.0]
    try:
        popt, pcov = curve_fit(logistic_5pl, time_vals, signal_vals, p0=p0, maxfev=10000)
        A, D, C, B, G = popt
        # Compute fitted curve
        x_fit = np.linspace(time_vals.min(), time_vals.max(), 200)
        y_fit = logistic_5pl(x_fit, *popt)
        # Estimate time to threshold
        t_thresh = inverse_5pl(threshold, *popt)

        st.subheader('Fit & Prediction')
        st.write(f'**Estimated time to threshold ({threshold}):** {t_thresh:.3f} (same units as time)')

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_vals, signal_vals, 'ko', label='Data')
        ax.plot(x_fit, y_fit, 'b-', label='5PL Fit')
        ax.axhline(threshold, color='green', linestyle='--', label='Threshold')
        ax.axvline(t_thresh, color='red', linestyle='--', label='Estimated Tt')
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal')
        ax.set_title('Real-Time 5PL Fit & Time-to-Threshold')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f'⚠️ 5PL fit failed: {e}')
else:
    st.info('Enter at least 5 data points to perform a 5PL fit.')

# ---- Download current data ----
if not st.session_state.df.empty:
    csv = st.session_state.df.to_csv(index=False).encode()
    st.download_button('Download Data as CSV', csv, file_name='rt_data.csv', mime='text/csv')
