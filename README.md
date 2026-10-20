# Loan application screening demo

This educational project trains a model to estimate **historical loan rejection likelihood** from the supplied CSV. The CSV records approvals and rejections; it has no repayment outcomes, so the model does **not** predict borrower default and must not be used to approve or decline real applications.

## Data limitations

- The source dataset has 4,269 records; its provenance and currency are not verified. Monetary values are kept in the source dataset's original units; no fixed exchange-rate conversion is applied.
- The dataset contains `self_employed`, not an active job inquiry. The app and model use self-employment consistently.
- Education level is included as a model input and must be entered consistently. The available file has no sensitive-group labels suitable for a fairness audit.
- Risk bands are display groupings only. They are not validated underwriting thresholds or lending advice.
- The displayed profile summary is not a causal model explanation.

## Setup (PowerShell)

From the project folder:

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the model by opening `loan_risk_pipeline.ipynb` in Jupyter and running its code cell. It validates the source columns, compares models with stratified cross-validation on the training split, calibrates the selected model using training data only, evaluates once on a held-out split, and writes `models/loan_model.pkl` plus `data/processed/model_card.json`.

With OIDC authentication configured, start the dashboard:

```powershell
/$env:LOAN_APP_LOCAL_DEMO = "1"
.\venv\Scripts\python.exe -m streamlit run ui\app.py --server.address 127.0.0.1/

```

## Authentication and local demo mode

For a local demonstration without OIDC credentials, open PowerShell in the project folder and run:

```powershell
$env:LOAN_APP_LOCAL_DEMO = "1"
.\venv\Scripts\python.exe -m streamlit run ui\app.py --server.address 127.0.0.1
```

Local demo mode is accepted only when Streamlit is bound to `127.0.0.1`, and it does not save application history. Use fictional data. For shared or deployed use, configure OIDC sign-in and an explicit email allowlist as described below; without either configuration, the app stays closed.

1. Register an OIDC application with a provider such as Google. For local use, set its callback/redirect URL to `http://localhost:8501/oauth2callback`.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
3. Replace the client ID, client secret, cookie secret, and example email with your provider values and the accounts you authorize. Generate a strong cookie secret with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
4. Install the auth dependency with `.\venv\Scripts\python.exe -m pip install -r requirements.txt`, then run the dashboard without `LOAN_APP_LOCAL_DEMO`.

The app filters saved history by the signed-in email. `secrets.toml` is ignored by Git; never share or commit it. The SQLite history file itself is still unencrypted on disk, and exported CSVs are saved by the browser. Keep the project folder and downloads on a device you control, and use fictional data for this demo. Existing history entries created before sign-in have no account owner and are hidden. To erase all local saved history, stop the app and remove `data/application_history*.db` after deciding you no longer need those records.

The app accepts inputs within the source data's observed ranges for income, loan amount, term, dependents, and credit score. Use compatible source-unit values. Retrain the model after changing the dataset or pipeline.

The dashboard also estimates monthly repayment, total interest, and total repayment. Enter the lender's annual interest rate, choose reducing-balance or flat-rate calculation, and optionally add one-time fees. The term determines the number of payments; it does not determine the interest rate. Monthly installments exclude one-time fees, while estimated total repayment includes them. These calculations are examples and may differ from a lender's contract.

The app opens on the **Application details** page. After the required fields are entered, **Analyze application** opens the dashboard. Use **Back to application details** to return with the submitted values preserved, or **New application / Clear fields** to start over.

## Local history and reports

Assessment CSV downloads are created in the browser. If you choose to save a record, the app writes applicant financial details and repayment estimates to `data/application_history_v4.db`, associated with the signed-in email. History is filtered by account in the app and can be deleted from the dashboard. The SQLite file is still unencrypted; anyone with access to the computer or project folder may be able to read it. Do not use real customer information or share the database or report files. Stop the app before deleting local history database files.

## Improving real-world validity

To predict repayment default, obtain appropriately governed loan-level data that includes repayment outcomes and timestamps. Split evaluation by time (and, where needed, by borrower) so the same borrower or later information cannot leak across train/test. Validate probability calibration, error costs, subgroup performance, and lending rules with qualified reviewers before any real-world use.
