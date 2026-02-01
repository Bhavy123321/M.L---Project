from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "loan_model.pkl"
model = joblib.load(MODEL_PATH)

EDUCATION_OPTIONS = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT_OPTIONS = ["Full-time", "Part-time", "Self-employed", "Unemployed"]


def to_float(value, field_name):
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            raise ValueError
        return v
    except Exception:
        raise ValueError(f"Invalid value for {field_name}")


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        education_options=EDUCATION_OPTIONS,
        employment_options=EMPLOYMENT_OPTIONS,
    )


@app.route("/version", methods=["GET"])
def version():
    # Use this to confirm Railway is running latest code
    return {
        "status": "ok",
        "code_version": "final-ui-pipeline-dataframe",
        "file": __file__,
    }


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = to_float(request.form.get("Age", ""), "Age")
        income = to_float(request.form.get("Income", ""), "Income")
        loan_amount = to_float(request.form.get("LoanAmount", ""), "LoanAmount")
        credit_score = to_float(request.form.get("CreditScore", ""), "CreditScore")
        dti = to_float(request.form.get("DTIRatio", ""), "DTIRatio")

        education = (request.form.get("Education") or "").strip()
        employment = (request.form.get("EmploymentType") or "").strip()

        if education not in EDUCATION_OPTIONS:
            raise ValueError("Please select a valid Education.")
        if employment not in EMPLOYMENT_OPTIONS:
            raise ValueError("Please select a valid Employment Type.")

        # sanity checks
        if age <= 0 or age > 100:
            raise ValueError("Age should be between 1 and 100.")
        if income < 0:
            raise ValueError("Income cannot be negative.")
        if loan_amount <= 0:
            raise ValueError("Loan Amount must be greater than 0.")
        if credit_score < 0 or credit_score > 1000:
            raise ValueError("Credit Score looks invalid (0–1000 expected).")
        if dti < 0 or dti > 2:
            raise ValueError("DTI Ratio looks invalid (0–2 typical). Example: 0.35")

        # ✅ IMPORTANT FIX: Pipeline expects DataFrame with exact column names
        X = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "DTIRatio": dti,
            "Education": education,
            "EmploymentType": employment
        }])

        # Predict
        pred = int(model.predict(X)[0])

        # Probability (if available)
        try:
            proba = float(model.predict_proba(X)[0][1])  # probability of class "1"
        except Exception:
            proba = None

        # UI labels
        status = "Approved ✅" if pred == 0 else "Rejected ❌"

        confidence = None
        if proba is not None:
            confidence = round((1 - proba) * 100, 2) if pred == 0 else round(proba * 100, 2)

        # UX hints
        hints = []
        if credit_score < 650:
            hints.append("Low Credit Score")
        if dti > 0.45:
            hints.append("High DTI Ratio")
        if income > 0 and loan_amount > (income * 0.6):
            hints.append("Loan Amount high vs Income")

        return render_template(
            "result.html",
            status=status,
            pred=pred,
            proba=None if proba is None else round(proba * 100, 2),
            confidence=confidence,
            age=age,
            income=income,
            loan_amount=loan_amount,
            credit_score=credit_score,
            dti=dti,
            education=education,
            employment=employment,
            hints=hints,
        )

    except Exception as e:
        return render_template(
            "index.html",
            education_options=EDUCATION_OPTIONS,
            employment_options=EMPLOYMENT_OPTIONS,
            error=str(e),
            form=request.form,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
