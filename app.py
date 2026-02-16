from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)

# ============ Model ============
MODEL_PATH = "loan_model.pkl"
model = joblib.load(MODEL_PATH)

EDUCATION_OPTIONS = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT_OPTIONS = ["Full-time", "Part-time", "Self-employed", "Unemployed"]

# ✅ Put your real URLs
SOCIAL_LINKS = {
    "linkedin": "https://www.linkedin.com/in/bhavy-soni-6123a32b0/",
    "github": "https://github.com/Bhavy123321"
}

# ✅ In-memory reviews (works fine for demo/projects)
REVIEWS = [
]

PROJECT_STATS = [
    {"label": "ML Pipeline", "value": "scikit-learn"},
    {"label": "Deployment", "value": "Railway"},
    {"label": "UI Style", "value": "Premium Glass"},
    {"label": "Pages", "value": "Predict • About • Reviews • Dashboard"},
]

# ============ Dashboard storage ============
PRED_FILE = "predictions.json"


def load_predictions():
    if not os.path.exists(PRED_FILE):
        return []
    try:
        with open(PRED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_prediction(entry: dict):
    data = load_predictions()
    data.append(entry)
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    return dict(
        social=SOCIAL_LINKS,
        brand_name="LoanSense"
    )


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        education_options=EDUCATION_OPTIONS,
        employment_options=EMPLOYMENT_OPTIONS,
    )


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", stats=PROJECT_STATS)


# ✅ FIXED: Single safe route for GET+POST
@app.route("/reviews", methods=["GET", "POST"])
def reviews_page():
    try:
        if request.method == "POST":
            name = (request.form.get("name") or "").strip()[:40] or "Anonymous"
            tag = (request.form.get("tag") or "").strip()[:30] or "User"
            message = (request.form.get("message") or "").strip()[:400] or "Great project!"
            rating_raw = (request.form.get("rating") or "5").strip()

            try:
                rating = int(rating_raw)
            except Exception:
                rating = 5

            rating = max(1, min(5, rating))

            REVIEWS.append({
                "name": name,
                "rating": rating,
                "message": message,
                "tag": tag,
                "date": datetime.now().strftime("%Y-%m-%d")
            })

            return redirect(url_for("reviews_page"))

        # GET
        latest = list(reversed(REVIEWS))
        total = len(REVIEWS)
        avg = round(sum(r["rating"] for r in REVIEWS) / total, 1) if total > 0 else 0

        return render_template(
            "reviews.html",
            reviews=latest,
            total_reviews=total,
            avg_rating=avg,
            page_error=None
        )

    except Exception as e:
        latest = list(reversed(REVIEWS))
        total = len(REVIEWS)
        avg = round(sum(r["rating"] for r in REVIEWS) / total, 1) if total > 0 else 0

        return render_template(
            "reviews.html",
            reviews=latest,
            total_reviews=total,
            avg_rating=avg,
            page_error=str(e)
        ), 200


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

        # ✅ STRICT: only positive real numbers allowed in ALL numeric fields
        if age <= 0 or age > 100:
            raise ValueError("Age should be between 1 and 100 (positive only).")
        if income <= 0:
            raise ValueError("Income must be greater than 0 (positive only).")
        if loan_amount <= 0:
            raise ValueError("Loan Amount must be greater than 0 (positive only).")
        if credit_score <= 0 or credit_score > 1000:
            raise ValueError("Credit Score looks invalid (1–1000 expected).")
        if dti <= 0 or dti > 2:
            raise ValueError("DTI Ratio looks invalid (must be > 0 and <= 2). Example: 0.35")

        # ✅ DataFrame with exact feature names
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

        # Your project mapping: pred==0 Approved
        status = "Approved ✅" if pred == 0 else "Rejected ❌"
        result_label = "Approved" if pred == 0 else "Rejected"

        confidence = None
        risk_pct = None
        safe_pct = None
        if proba is not None:
            risk_pct = round(proba * 100, 2)
            safe_pct = round((1 - proba) * 100, 2)
            confidence = safe_pct if pred == 0 else risk_pct

        hints = []
        if credit_score < 650:
            hints.append("Low Credit Score")
        if dti > 0.45:
            hints.append("High DTI Ratio")
        if income > 0 and loan_amount > (income * 0.6):
            hints.append("Loan Amount high vs Income")
        if employment == "Unemployed":
            hints.append("Employment stability risk")

        # ✅ SAVE for dashboard counts + charts
        save_prediction({
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Age": round(float(age), 4),
            "Income": round(float(income), 4),
            "LoanAmount": round(float(loan_amount), 4),
            "CreditScore": round(float(credit_score), 4),
            "DTIRatio": round(float(dti), 6),
            "Education": education,
            "EmploymentType": employment,
            "pred": pred,
            "result": result_label,
            "probability": None if proba is None else round(proba * 100, 2),
        })

        return render_template(
            "result.html",
            status=status,
            pred=pred,
            proba=None if proba is None else round(proba * 100, 2),
            confidence=confidence,
            risk_pct=risk_pct,
            safe_pct=safe_pct,
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


# ✅ NEW: Dashboard route
@app.route("/dashboard", methods=["GET"])
def dashboard():
    data = load_predictions()

    total = len(data)
    approved = sum(1 for x in data if x.get("result") == "Approved")
    rejected = sum(1 for x in data if x.get("result") == "Rejected")

    # Trend by day (YYYY-MM-DD)
    daily = defaultdict(int)
    for x in data:
        d = (x.get("created_at", "")[:10]).strip()
        if d:
            daily[d] += 1

    trend_labels = sorted(daily.keys())
    trend_counts = [daily[d] for d in trend_labels]

    recent = list(reversed(data))[:10]

    return render_template(
        "dashboard.html",
        total=total,
        approved=approved,
        rejected=rejected,
        trend_labels=trend_labels,
        trend_counts=trend_counts,
        recent=recent,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

