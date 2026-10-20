# AI-Based Loan Application Prediction System

An educational machine learning project that estimates historical loan rejection likelihood using a dataset of 4,269 loan application records.

## Features
- Machine learning model training and evaluation.
- Interactive loan application form using Streamlit.
- Historical rejection likelihood predictions.
- Estimated monthly installments, interest, and total repayment.
- Optional local assessment history and CSV reports.
- Optional OIDC authentication.

## Technologies
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- SQLite
- Jupyter Notebook

## Installation

```powershell
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the Model

Open `loan_risk_pipeline.ipynb` in Jupyter and run the training cells.

## Run the Application

```powershell
$env:LOAN_APP_LOCAL_DEMO = "1"
.\venv\Scripts\python.exe -m streamlit run ui/app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501 in your browser.

## Disclaimer

This project is for educational purposes only. It estimates historical loan rejection likelihood, not borrower default. It is not an official bank service and must not be used to approve or reject real loan applications.

**Data limitation:** The dataset's provenance and currency are unverified. Monetary values retain their original units.
