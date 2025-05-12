Real-Time Time-to-Threshold Calculator

A Streamlit application for entering time and sensor signal data in real time and estimating the time needed to reach a desired threshold using a Five-Parameter Logistic (5PL) model.

Features

Real-time data entry: Input time and sensor readings as they come in.

5PL Curve Fit: Automatically fits a 5PL model once you have ≥5 data points.

Time-to-Threshold Prediction: Calculates and displays the estimated time to reach your specified threshold.

Interactive Plot: Shows raw data (black circles), 5PL fit (blue solid line), 95% CI bands (red shaded area), threshold line (green dashed), and predicted Tt marker.

CSV Export: Download your collected dataset at any time.

Getting Started (Local)

Prerequisites

Python 3.7 or higher

Git (to clone the repo)

Installation

Clone the repository:

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

Install dependencies:

pip install -r requirements.txt

Running Locally

streamlit run real_time_ttt_app.py

Open http://localhost:8501 in your browser.

Deployment to Streamlit Community Cloud

Push your code (including real_time_ttt_app.py and requirements.txt) to a public GitHub repository.

Go to https://share.streamlit.io and sign in with your GitHub account.

Click “New app”, select your GitHub repo, branch, and the path to real_time_ttt_app.py.

Click “Deploy”. Your app will be live at https://share.streamlit.io/<username>/<repo>/<path>.

File Structure

├── real_time_ttt_app.py    # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # This file



Usage

Enter numerical Time and Signal values and click “Add Data Point”.

Once ≥5 points are in the table, the app fits a 5PL curve.

Set your Threshold. The plot updates with the threshold line and estimated Tt.

Download your data with the “Download Data as CSV” button.

License

This project is licensed under the MIT License.

