from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "loan_model.pkl"
model = joblib.load(MODEL_PATH)

EDUCATION_OPTIONS = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT_OPTIONS = ["Full-time", "Part-time", "Self-employed", "Unemployed"]

# ✅ Put your real links here
SOCIAL_LINKS = {
    "linkedin": "https://www.linkedin.com/in/bhavy-soni-6123a32b0/",
    "github": "https://github.com/Bhavy123321"
}

# Simple in-memory reviews (resets when app restarts)
REVIEWS = [
    {
        "name": "Aarav",
        "rating": 5,
        "message": "Clean UI and super fast prediction. Loved it!",
        "tag": "Student"
    },
    {
        "name": "Neha",
        "rating": 4,
        "message": "Very smooth experience. Looks professional.",
        "tag": "Developer"
    }
]


def to_float(value, field_name):
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            raise ValueError
        return v
    except Exception:
        raise ValueError(f"Invalid value for {field_name}")


@app.context_processor
def inject_globals():
    # Available in all templates automatically
    return dict(social=SOCIAL_LINKS)


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        education_options=EDUCATION_OPTIONS,
        employment_options=EMPLOYMENT_OPTIONS,
    )


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/reviews", methods=["GET"])
def reviews():
    # show latest first
    latest = list(reversed(REVIEWS))
    return render_template("reviews.html", reviews=latest)


@app.route("/reviews", methods=["POST"])
def add_review():
    name = (request.form.get("name") or "").strip()[:40]
    tag = (request.form.get("tag") or "").strip()[:30]
    message = (request.form.get("message") or "").strip()[:300]
    rating_raw = (request.form.get("rating") or "").strip()

    try:
        rating = int(rating_raw)
    except Exception:
        rating = 5

    if not name:
        name = "Anonymous"
    if not tag:
        tag = "User"
    if not message:
        message = "Great project!"

    rating = max(1, min(5, rating))

    REVIEWS.append({
        "name": name,
        "rating": rating,
        "message": message,
        "tag": tag
    })

    return redirect(url_for("reviews"))


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

        # ✅ DataFrame with exact column names
        X = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "DTIRatio": dti,
            "Education": education,
            "EmploymentType": employment
        }])

        pred = int(model.predict(X)[0])

        try:
            proba = float(model.predict_proba(X)[0][1])
        except Exception:
            proba = None

        status = "Approved ✅" if pred == 0 else "Rejected ❌"

        confidence = None
        if proba is not None:
            confidence = round((1 - proba) * 100, 2) if pred == 0 else round(proba * 100, 2)

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


