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
