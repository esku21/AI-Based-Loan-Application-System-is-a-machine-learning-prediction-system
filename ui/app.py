import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import sqlite3
import math
from pathlib import Path
from datetime import datetime, timezone

# --- PATH HANDLER ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(parent_dir, 'models', 'loan_model.pkl')
HISTORY_PATH = os.path.join(parent_dir, 'data', 'application_history_v4.db')
CBE_LOGO_URL = "https://pbs.twimg.com/media/GAVebfwXAAASd-e.jpg"
MODEL_FEATURES = [
    "credit_score",
    "debt_to_income",
    "asset_coverage",
    "loan_term",
    "dependents",
    "is_self_employed",
    "is_graduate",
]

# --- STYLE ENGINE ---
st.set_page_config(page_title="AI Based Loan Approval Prediction System", page_icon="🏦", layout="wide")

# CSS stays separate from the prediction logic and adapts on mobile screens.
st.html(Path(__file__).with_name("style.css"))

# OIDC authentication is required before any applicant data or model is loaded.
def require_authorized_user():
    """Require configured OpenID Connect sign-in and an explicit email allowlist."""
    try:
        auth_config = st.secrets.get("auth", {})
        privacy_config = st.secrets.get("privacy", {})
    except Exception:
        auth_config, privacy_config = {}, {}
    allowed_emails = {
        str(email).strip().lower()
        for email in privacy_config.get("allowed_emails", [])
        if str(email).strip()
    }

    required_auth_keys = {"redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url"}
    if not required_auth_keys.issubset(auth_config) or not allowed_emails:
        local_demo_requested = os.environ.get("LOAN_APP_LOCAL_DEMO") == "1"
        local_bind_addresses = {"127.0.0.1", "localhost", "::1"}
        bind_address = str(st.get_option("server.address") or "").strip().lower()
        if local_demo_requested and bind_address in local_bind_addresses:
            st.session_state["local_demo_mode"] = True
            return None

        st.title("Private access is not configured")
        st.error(
            "This app requires an OpenID Connect provider and an allowed email list "
            "before applicant information can be viewed. To run a no-history local demo, "
            "set LOAN_APP_LOCAL_DEMO=1 and bind Streamlit to 127.0.0.1. See the README "
            "for the PowerShell command."
        )
        st.stop()

    if not st.user.is_logged_in:
        st.title("Sign in required")
        st.info("Sign in with an account explicitly authorized to use this loan screening demo.")
        if st.button("Sign in", type="primary"):
            st.login()
        st.stop()

    email = str(st.user.get("email", "")).strip().lower()
    email_verified = st.user.get("email_verified", False) is True
    if not email or not email_verified or email not in allowed_emails:
        st.title("Access denied")
        st.error("Your account must provide a verified email address on this app's access list.")
        if st.button("Sign out"):
            st.logout()
        st.stop()
    return email


auth_user_email = require_authorized_user()

# --- BUSINESS ENGINE ---
def calculate_features(df):
    """Normalized feature set generation - Vectorized logic matching pipeline."""
    df_eng = df.copy()
    # Debt to Income Ratio
    df_eng['debt_to_income'] = df_eng['loan_amount'] / df_eng['income'].replace(0, 1)
    # Asset Coverage Ratio
    df_eng['asset_coverage'] = df_eng['total_assets'] / df_eng['loan_amount'].replace(0, 1)
    return df_eng

def process_analysis(probability, predicted_class):
    prediction = "Rejected" if predicted_class == 1 else "Approved"
    if probability is None:
        return {
            "tier": "Probability unavailable",
            "code": "N/A",
            "raw": None,
            "prediction": prediction,
        }
    if probability < 0.10: tier, tag = "A", "Lower estimated rejection likelihood"
    elif probability < 0.30: tier, tag = "B", "Moderate estimated rejection likelihood"
    elif probability < 0.60: tier, tag = "C", "Elevated estimated rejection likelihood"
    else: tier, tag = "D", "High estimated rejection likelihood"
    return {"tier": tag, "code": tier, "raw": probability, "prediction": prediction}

def calculate_repayment(principal, annual_rate_percent, years, method, upfront_fees=0):
    """Estimate monthly repayments; fees are paid once and excluded from installments."""
    months = years * 12
    annual_rate = annual_rate_percent / 100
    if method == "Reducing balance":
        monthly_rate = annual_rate / 12
        if monthly_rate == 0:
            monthly_payment = principal / months
        else:
            monthly_payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)
        total_interest = monthly_payment * months - principal
    else:
        total_interest = principal * annual_rate * years
        monthly_payment = (principal + total_interest) / months
    total_repayment = monthly_payment * months + upfront_fees
    return {
        "monthly_payment": monthly_payment,
        "total_interest": total_interest,
        "upfront_fees": upfront_fees,
        "total_repayment": total_repayment,
    }

def init_history():
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with sqlite3.connect(HISTORY_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            income REAL, loan_amount REAL, loan_term INTEGER, credit_score INTEGER,
            dependents INTEGER, self_employed TEXT, education TEXT, assets REAL,
            annual_interest_rate REAL, repayment_method TEXT, upfront_fees REAL,
            monthly_payment REAL, total_interest REAL, total_repayment REAL,
            probability REAL, status TEXT, tier TEXT
        )""")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(applications)")}
        if "owner_email" not in columns:
            conn.execute("ALTER TABLE applications ADD COLUMN owner_email TEXT NOT NULL DEFAULT ''")

def save_application(values, owner_email):
    init_history()
    with sqlite3.connect(HISTORY_PATH) as conn:
        conn.execute("""INSERT INTO applications
            (created_at, income, loan_amount, loan_term, credit_score, dependents,
             self_employed, education, assets, annual_interest_rate, repayment_method,
             upfront_fees, monthly_payment, total_interest, total_repayment,
             probability, status, tier, owner_email)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (*values, owner_email))

def load_history(owner_email):
    init_history()
    with sqlite3.connect(HISTORY_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM applications WHERE lower(owner_email) = lower(?) ORDER BY id DESC",
            conn,
            params=(owner_email,),
        )

def clear_history(owner_email):
    init_history()
    with sqlite3.connect(HISTORY_PATH) as conn:
        conn.execute("DELETE FROM applications WHERE lower(owner_email) = lower(?)", (owner_email,))

def clear_application_form():
    st.session_state["application_generation"] = st.session_state.get("application_generation", 0) + 1
    st.session_state.pop("last_submitted_inputs", None)
    st.session_state["restore_application_inputs"] = False
    st.session_state["app_page"] = "application"


def return_to_application():
    st.session_state["app_page"] = "application"
    st.session_state["restore_application_inputs"] = True

def build_explanations(inc, princ, score, hands, employed, education, probability):
    explanations = [
        f"Requested loan is {princ / inc:.1%} of annual income.",
        f"Credit score: {score}.",
        f"Applicant profile: {hands} dependents, self-employed: {employed}, education: {education}.",
    ]
    if probability is not None:
        explanations.append(
            f"The model estimates {probability:.2%} historical rejection likelihood. Summaries are not causal explanations."
        )
    else:
        explanations.append("The model does not provide class probabilities. This summary is not a causal explanation.")
    return explanations

# --- MODEL CONSUMER ---
@st.cache_resource
def get_ai_brain(model_path, modified_time):
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    return None

brain = get_ai_brain(MODEL_PATH, os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0)

# --- SIDEBAR ---
with st.sidebar:
    if auth_user_email is None:
        st.warning("Local demo only. Use fictional data; saving history is disabled.")
    else:
        st.caption(f"Signed in: {auth_user_email}")
        if st.button("Sign out", width="stretch"):
            st.logout()

# --- PROCESSING PIPELINE ---
def generate_inference(inc, princ, yrs, score, hands, employed, education, assets):
    if brain is None: return None, None
    
    payload = pd.DataFrame([{
        'income': inc, 'loan_amount': princ, 'loan_term': yrs,
        'credit_score': score, 'dependents': hands,
        'is_self_employed': 1 if employed == "Yes" else 0,
        'is_graduate': 1 if education == "Graduate" else 0,
        'total_assets': assets
    }])
    
    try:
        processed = calculate_features(payload)
        X = processed.loc[:, MODEL_FEATURES].astype("float64")
        expected_features = getattr(brain, "feature_names_in_", None)
        if expected_features is not None and list(expected_features) != MODEL_FEATURES:
            raise ValueError(
                "The trained model's input feature names or order do not match the application pipeline."
            )
        predicted_class = int(brain.predict(X)[0])
        prob = None
        if callable(getattr(brain, "predict_proba", None)):
            classes = list(getattr(brain, "classes_", []))
            if 1 not in classes:
                raise ValueError("The trained model does not contain the rejection class (1).")
            prob = float(brain.predict_proba(X)[0][classes.index(1)])
        return prob, process_analysis(prob, predicted_class)
    except Exception as e:
        return 0, {"status": "ERROR", "note": f"Pipeline failure: {e}"}

# --- APPLICATION PAGE ---
st.session_state.setdefault("app_page", "application")
submitted = False
if st.session_state["app_page"] != "dashboard":
    with st.container(horizontal_alignment="center"):
        try:
            st.image(CBE_LOGO_URL, width=180, alt="Commercial Bank of Ethiopia emblem and wordmark")
        except Exception:
            st.markdown("### CBE", text_alignment="center")
    st.title("Loan application details", text_alignment="center")
    st.caption("Complete the form, then analyze the application to open the dashboard.", text_alignment="center")
    st.caption("Educational prototype only. This is not an official CBE service or a real lending application.", text_alignment="center")
    st.info("Enter all monetary amounts in the same currency. The source dataset's currency is unverified.", icon=":material/info:")

    form_generation = st.session_state.get("application_generation", 0)
    if st.session_state.pop("restore_application_inputs", False):
        saved = st.session_state.get("last_submitted_inputs", {})
        form_keys = {
            "income": "annual_income", "principal": "requested_loan", "term": "loan_term",
            "credit_score": "credit_score", "hands": "dependents", "employed": "self_employed",
            "education": "education", "assets": "assets", "annual_rate": "annual_rate",
            "repayment_method": "repayment_method", "upfront_fees": "upfront_fees",
        }
        for input_name, widget_name in form_keys.items():
            if input_name in saved:
                st.session_state[f"{widget_name}_{form_generation}"] = saved[input_name]

    with st.form("loan_application_form"):
        left_input, right_input = st.columns(2, wrap=True)
        with left_input:
            income = st.number_input("Annual income", min_value=1, max_value=100000000, value=None, step=10000, key=f"annual_income_{form_generation}")
            principal = st.number_input("Requested loan amount", min_value=1, max_value=100000000, value=None, step=5000, key=f"requested_loan_{form_generation}")
            term = st.slider("Loan term (years)", 2, 20, 10, key=f"loan_term_{form_generation}")
            credit_score = st.slider("Credit score", 300, 900, 600, key=f"credit_score_{form_generation}")
            hands = st.number_input("Family members supported by applicant", 0, 5, 2, key=f"dependents_{form_generation}", help="Count people who rely on the applicant's income. Enter 0 if nobody depends on it.")
        with right_input:
            employed = st.radio("Self-employed?", ["Yes", "No"], horizontal=True, key=f"self_employed_{form_generation}")
            education = st.radio("Education level", ["Graduate", "Not graduate"], horizontal=True, key=f"education_{form_generation}")
            assets = st.number_input("Total asset value", min_value=0, max_value=200000000, value=None, step=10000, key=f"assets_{form_generation}")
            annual_rate = st.number_input("Annual interest rate (%)", min_value=0.0, max_value=100.0, value=None, step=0.25, format="%.2f", key=f"annual_rate_{form_generation}", help="Enter the annual rate offered by the lender. The app does not determine a rate from the loan term.")
            repayment_method = st.radio("Interest calculation method", ["Reducing balance", "Flat rate"], key=f"repayment_method_{form_generation}", help="Reducing balance charges interest on the remaining principal. Flat rate calculates interest on the original principal for the full term.")
            upfront_fees = st.number_input("One-time fees (optional)", min_value=0.0, max_value=100000000.0, value=0.0, step=100.0, key=f"upfront_fees_{form_generation}", help="These fees are included in total repayment, but not in the monthly installment.")
        submitted = st.form_submit_button("Analyze application", type="primary", width="stretch")
        st.form_submit_button("New application / Clear fields", on_click=clear_application_form, width="stretch")

    if submitted:
        missing_values = [name for name, value in {
            "Annual income": income,
            "Requested loan amount": principal,
            "Total asset value": assets,
            "Annual interest rate": annual_rate,
        }.items() if value is None]
        if missing_values:
            st.error("Enter values for: " + ", ".join(missing_values) + ".")
        else:
            st.session_state["last_submitted_inputs"] = {
                "income": income, "principal": principal, "term": term,
                "credit_score": credit_score, "hands": hands, "employed": employed,
                "education": education, "assets": assets, "annual_rate": annual_rate,
                "repayment_method": repayment_method, "upfront_fees": upfront_fees,
            }
            st.session_state["app_page"] = "dashboard"
            st.rerun()
    st.stop()

# --- MAIN DASHBOARD ---
st.button(":material/arrow_back: Back to application details", on_click=return_to_application)
st.title("Loan application dashboard")
st.caption("Predicts the historical approval/rejection label from the sample dataset. It does not predict repayment or make a real lending decision.")

if brain is None:
    st.warning("⚠️ **Initialization Required**: No trained AI model brain detected. Please run the `loan_risk_pipeline.ipynb` to generate the system file first.")
    st.stop()

if "last_submitted_inputs" not in st.session_state:
    st.session_state["app_page"] = "application"
    st.rerun()

income = st.session_state["last_submitted_inputs"]["income"]
principal = st.session_state["last_submitted_inputs"]["principal"]
term = st.session_state["last_submitted_inputs"]["term"]
credit_score = st.session_state["last_submitted_inputs"]["credit_score"]
hands = st.session_state["last_submitted_inputs"]["hands"]
employed = st.session_state["last_submitted_inputs"]["employed"]
education = st.session_state["last_submitted_inputs"]["education"]
assets = st.session_state["last_submitted_inputs"]["assets"]
annual_rate = st.session_state["last_submitted_inputs"]["annual_rate"]
repayment_method = st.session_state["last_submitted_inputs"]["repayment_method"]
upfront_fees = st.session_state["last_submitted_inputs"]["upfront_fees"]

prob, decision = generate_inference(income, principal, term, credit_score, hands, employed, education, assets)
if decision is None or decision.get('status') == 'ERROR':
    st.error(decision.get('note', 'Could not generate a prediction.') if decision else 'Could not generate a prediction.')
    st.stop()
repayment = calculate_repayment(principal, annual_rate, term, repayment_method, upfront_fees)
def show_probability_chart(probability, bar_color):
    if probability is None:
        st.info("The loaded model does not provide class probabilities, so no probability chart is shown.")
        return

    st.markdown("### Probability analysis")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Estimated rejection likelihood (%)", "font": {"size": 18}},
        gauge={
            "axis": {
                "range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue",
                "tickmode": "array", "tickvals": [0, 10, 30, 60, 100],
                "ticktext": ["0%", "10%", "30%", "60%", "100%"],
            },
            "bar": {"color": bar_color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 10], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [10, 30], "color": "rgba(251, 191, 36, 0.2)"},
                {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
    ))

    hover_bands = [
        (0, 10, "#10b981", "0?10%: Lower estimated rejection likelihood. This is descriptive, not an approval rule."),
        (10, 30, "#fbbf24", "10?30%: Moderate estimated rejection likelihood. This is descriptive, not an approval rule."),
        (30, 60, "#f59e0b", "30?60%: Elevated estimated rejection likelihood. This is descriptive, not an approval rule."),
        (60, 100, "#ef4444", "60?100%: High estimated rejection likelihood. This is descriptive, not an approval rule."),
    ]
    for lower, upper, color, description in hover_bands:
        values = [lower + (upper - lower) * point / 40 for point in range(41)]
        angles = [math.pi * (1 - value / 100) for value in values]
        xs = [0.5 + 0.43 * math.cos(angle) for angle in angles]
        ys = [0.08 + 0.68 * math.sin(angle) for angle in angles]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line={"color": color, "width": 28},
            opacity=0.01, hovertemplate=f"{description}<extra></extra>",
            showlegend=False, name=description.split(":")[0],
        ))

    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", hovermode="closest",
        xaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        yaxis={"range": [0, 1], "visible": False, "fixedrange": True},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Probability bands are illustrative display ranges, not validated lending thresholds.")


c1, c2 = st.columns([1.5, 1], wrap=True)
chart_color = "#f59e0b" if decision["code"] in ("B", "C") else "#64748b"

with c1:
    with st.container(border=True):
        status_color = "green" if decision["prediction"] == "Approved" else "red"
        st.markdown(
            f"#### Model prediction: :{status_color}[**{decision['prediction'].upper()}**]"
        )
        if prob is None:
            st.caption("Predicted historical label | The model does not provide class probabilities.")
        else:
            st.caption(f"Predicted historical label | Estimated rejection likelihood: {prob:.1%}")
        st.info("This reproduces labels in the training dataset. It is not an actual approval or rejection decision.")
        if credit_score <= 550:
            st.warning(
                "Training-data context: 89.4% of records with a credit score of 550 or below were labeled rejected. "
                "This may be why changing only the loan amount does not change the predicted label. Enter the applicant's "
                "verified details; do not adjust values to force a particular result."
            )

    if prob is None:
        m1, m2 = st.columns(2, wrap=True)
        m1.metric("Predicted historical label", decision["prediction"])
        m2.metric("Requested amount", f"{principal:,.0f}")
    else:
        m1, m2, m3 = st.columns(3, wrap=True)
        m1.metric("Estimated rejection likelihood", f"{prob:.2%}")
        m2.metric("Screening band", decision["code"])
        m3.metric("Requested amount", f"{principal:,.0f}")

with c2:
    show_probability_chart(prob, chart_color)

st.divider()

col_metrics1, col_metrics2 = st.columns(2, wrap=True)

with col_metrics1:
    st.subheader("Income ratio")
    dti = principal / income if income > 0 else 1
    st.write(f"**DTI (Cash Flow Ratio):** `{dti:.1%}`")
    st.progress(min(dti, 1.0))
    st.caption("Descriptive ratio only; no lending threshold is configured.")

with col_metrics2:
    st.subheader("Asset coverage")
    cov = assets / principal if principal > 0 else 0
    st.write(f"**Collateral Coverage:** `{cov:.1f}x`")
    st.progress(min(cov/5.0, 1.0))
    st.caption("Descriptive ratio only; no lending threshold is configured.")

st.subheader("Estimated repayment")
pay1, pay2, pay3, pay4 = st.columns(4, wrap=True)
pay1.metric("Estimated monthly payment", f"{repayment['monthly_payment']:,.2f}")
pay2.metric("Estimated total interest", f"{repayment['total_interest']:,.2f}")
pay3.metric("One-time fees", f"{repayment['upfront_fees']:,.2f}")
pay4.metric("Estimated total repayment", f"{repayment['total_repayment']:,.2f}")
st.caption(f"{repayment_method} estimate at {annual_rate:.2f}% annual interest over {term} years. Monthly payments exclude one-time fees; total repayment includes them. Actual lender schedules and charges may differ.")

st.subheader("Applicant profile summary")
st.caption("These details describe the submitted profile and model score; they do not establish why the model made its prediction.")
for factor in build_explanations(income, principal, credit_score, hands, employed, education, prob):
    st.markdown(f"- {factor}")

report = pd.DataFrame([{
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "annual_income_same_currency": income, "requested_loan_same_currency": principal,
    "term_years": term, "credit_score": credit_score, "dependents": hands,
    "self_employed": employed, "education_level": education, "assets_same_currency": assets,
    "annual_interest_rate_percent": annual_rate, "repayment_method": repayment_method,
    "upfront_fees_same_currency": upfront_fees,
    "estimated_monthly_payment": repayment["monthly_payment"],
    "estimated_total_interest": repayment["total_interest"],
    "estimated_total_repayment": repayment["total_repayment"],
    "estimated_rejection_likelihood": prob, "screening_band": decision["code"], "predicted_historical_label": decision["prediction"],
}])
left, right = st.columns(2, wrap=True)
left.download_button("Download assessment CSV", report.to_csv(index=False).encode("utf-8-sig"),
                     file_name="loan_assessment.csv", mime="text/csv")
st.caption("Assessment downloads contain financial details. Save them only to a device and folder you control.")
if auth_user_email is not None and right.button("Save assessment to history"):
    save_application((datetime.now(timezone.utc).isoformat(timespec="seconds"), income, principal,
                      term, credit_score, hands, employed, education, assets,
                      annual_rate, repayment_method, upfront_fees,
                      repayment["monthly_payment"], repayment["total_interest"],
                      repayment["total_repayment"], prob, decision["tier"], decision["code"]), auth_user_email)
    st.success("Assessment saved locally.")

if auth_user_email is None:
    st.info("Local demo mode does not save application history. Use fictional data only.")
else:
    with st.expander("Application history and portfolio summary"):
        history = load_history(auth_user_email)
        if history.empty:
            st.caption("No saved assessments for your account. Use 'Save assessment to history' to add one.")
        else:
            a, b, c = st.columns(3, wrap=True)
            a.metric("Saved applications", len(history))
            b.metric("Band A", int((history["tier"] == "A").sum()))
            c.metric("Band D", int((history["tier"] == "D").sum()))
            st.dataframe(history, width="stretch", hide_index=True)
            st.download_button("Download history CSV", history.to_csv(index=False).encode("utf-8-sig"),
                               file_name="loan_application_history.csv", mime="text/csv")
            st.caption("History is stored locally without database encryption and is visible only to your authorized account in this app.")
            if st.checkbox("I understand this permanently deletes my saved assessment history", key="confirm_history_delete"):
                if st.button("Delete all saved history", type="secondary"):
                    clear_history(auth_user_email)
                    st.success("Your saved history was deleted. Rerun the app to refresh the table.")

st.warning("Training data contains past approvals/rejections only. The output is a dataset label prediction, not a real lending decision or default prediction.")
