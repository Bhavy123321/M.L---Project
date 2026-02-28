from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
import joblib
import pandas as pd

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")

app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------------------------------
# GLOBAL LINKS
# -------------------------------------------------
@app.context_processor
def inject_globals():
    return {
        "brand_name": "Loan Default Predictor",
        "social": {
            "linkedin": "https://www.linkedin.com/in/bhavy-soni-6123a32b0/",
            "github": "https://github.com/Bhavy123321",
            "instagram": "#",
            "twitter": "#",
        },
    }

# -------------------------------------------------
# DATABASE INIT
# -------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age REAL,
                income REAL,
                loan_amount REAL,
                credit_score REAL,
                dti REAL,
                education TEXT,
                employment_type TEXT,
                prediction INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                tag TEXT,
                rating INTEGER NOT NULL,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

init_db()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
_model = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError("loan_model.pkl not found.")
    return _model

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
@app.route("/")
def dashboard():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT 
                created_at,
                age AS Age,
                income AS Income,
                loan_amount AS LoanAmount,
                credit_score AS CreditScore,
                dti AS DTIRatio,
                prediction,
                CASE WHEN prediction = 0 THEN 'Approved' ELSE 'Rejected' END as result
            FROM history
            ORDER BY id DESC
        """, conn)

    total = len(df)
    approved = int((df["prediction"] == 0).sum()) if total else 0
    rejected = int((df["prediction"] == 1).sum()) if total else 0

    return render_template(
        "dashboard.html",
        total=total,
        approved=approved,
        rejected=rejected,
        recent=df.to_dict(orient="records"),
        trend_labels=[],
        trend_counts=[]
    )

# -------------------------------------------------
# PREDICT
# -------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("index.html")

    try:
        model = get_model()

        age = float(request.form["Age"])
        income = float(request.form["Income"])
        loan_amt = float(request.form["LoanAmount"])
        credit = float(request.form["CreditScore"])
        dti = float(request.form["DTIRatio"])
        edu = request.form["Education"]
        emp = request.form["EmploymentType"]

        X = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amt,
            "CreditScore": credit,
            "DTIRatio": dti,
            "Education": edu,
            "EmploymentType": emp
        }])

        pred = int(model.predict(X)[0])

        safe_prob = 0
        risk_prob = 0

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            safe_prob = round(proba[0] * 100, 2)
            risk_prob = round(proba[1] * 100, 2)
        else:
            safe_prob = 100 if pred == 0 else 0
            risk_prob = 100 if pred == 1 else 0

        confidence = max(safe_prob, risk_prob)

        # Save prediction
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO history (
                    age, income, loan_amount, credit_score, dti,
                    education, employment_type, prediction
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (age, income, loan_amt, credit, dti, edu, emp, pred))
            conn.commit()

        status = "Loan Approved ✅" if pred == 0 else "Loan Rejected ❌"

        return render_template(
            "result.html",
            result=status,
            prediction=pred,
            safe_prob=safe_prob,
            risk_prob=risk_prob,
            confidence=confidence
        )

    except Exception as e:
        return render_template("index.html", error=str(e))

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
