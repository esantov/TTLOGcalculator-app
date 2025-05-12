Real-Time Time-to-Threshold Calculator

A Streamlit application for entering time and sensor signal data in real time and estimating the time required to reach a desired threshold using a Five-Parameter Logistic (5PL) model.

Features

Real-time data entry: Input time and sensor readings as they arrive.

5PL Curve Fitting: Automatically fits a 5PL model once you have 5 or more data points.

Time-to-Threshold Prediction: Calculates and displays the estimated time to reach a specified threshold.

Interactive Plot: Shows raw data (black circles), fitted curve (blue solid line), 95% CI bands (red shaded area), threshold line (green dashed), and predicted Tt marker.

CSV Export: Download the collected dataset anytime.

Streamlit Cloud Deployment

Prepare your GitHub repository:

Ensure your app file is named streamlit_app.py (or update on Streamlit settings).

Include a requirements.txt at the root listing all dependencies:

streamlit==1.30.0
pandas==2.0.3
numpy==1.25.2
scipy==1.11.1
matplotlib==3.8.0

Push your code to a public GitHub repository.

Navigate to Streamlit Community Cloud and sign in with GitHub.

Click “New app”, select your repo, branch, and the path to streamlit_app.py.

Click “Deploy”. Your app will build and run at:

https://share.streamlit.io/<your-username>/<repo-name>/streamlit_app.py

Troubleshooting Deployment Errors

ModuleNotFoundError:

In Manage App → Logs, look for ERROR: Could not find a version that satisfies the requirement ....

Update your requirements.txt to include the missing package (e.g., scipy).

Commit & push the change—Streamlit Cloud will rebuild with the corrected requirements.

Build Failures:

Confirm you can install your requirements locally:

pip install -r requirements.txt

Fix any typos in package names or version incompatibilities.

Push updates until the build log shows successful Collecting ... and Installing collected packages steps.

File Structure

├── streamlit_app.py    # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This documentation

Usage on Streamlit

After deployment:

Open the app URL in your browser.

Enter Time and Signal values, click “Add Data Point”.

Once at least 5 points are entered, the 5PL fit and Tt prediction appear.

Adjust Threshold as needed; the plot updates live.

Download your data via the “Download Data as CSV” button.

License

Distributed under the MIT License. See LICENSE for details.

