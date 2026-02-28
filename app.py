from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
import joblib
import pandas as pd

# -------------------------------------------------
# PATHS / APP CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")

app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------------------------------
# GLOBAL CONTEXT
# -------------------------------------------------
@app.context_processor
def inject_globals():
    return {
        "brand_name": "Loan Default",
        "social": {
            "linkedin": "#",
            "github": "#",
            "instagram": "#",
            "twitter": "#",
        },
    }

# -------------------------------------------------
# DATABASE HELPERS
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # History table (dashboard)
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

    # Reviews table
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
    conn.close()

init_db()

# -------------------------------------------------
# MODEL HELPERS
# -------------------------------------------------
_model = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return _model

def template_exists(name: str) -> bool:
    return os.path.exists(os.path.join(BASE_DIR, "templates", name))

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

# ✅ Dashboard
@app.route("/")
def dashboard():
    conn = sqlite3.connect(DB_PATH)

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
        LIMIT 50
    """, conn)

    trend_df = pd.read_sql_query("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM history
        GROUP BY DATE(created_at)
        ORDER BY date ASC
        LIMIT 10
    """, conn)

    conn.close()

    total = len(df)
    approved = int((df["prediction"] == 0).sum()) if total else 0
    rejected = int((df["prediction"] == 1).sum()) if total else 0

    recent_list = df.to_dict(orient="records")

    return render_template(
        "dashboard.html",
        total=total,
        approved=approved,
        rejected=rejected,
        recent=recent_list,
        trend_labels=trend_df["date"].tolist() if not trend_df.empty else [],
        trend_counts=trend_df["count"].tolist() if not trend_df.empty else []
    )

# ✅ Predictor
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        if template_exists("index.html"):
            return render_template("index.html")
        return "Predictor page (index.html) not found.", 404

    try:
        model = get_model()

        # Extract form data
        age = float(request.form.get("Age", 0))
        income = float(request.form.get("Income", 0))
        loan_amt = float(request.form.get("LoanAmount", 0))
        credit = float(request.form.get("CreditScore", 0))
        dti = float(request.form.get("DTIRatio", 0))
        edu = request.form.get("Education", "")
        emp = request.form.get("EmploymentType", "")

        X = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amt,
            "CreditScore": credit,
            "DTIRatio": dti,
            "Education": edu,
            "EmploymentType": emp
        }])

        # Prediction class (0/1)
        pred = int(model.predict(X)[0])

        # ✅ Probability + Confidence (fix for % showing blank)
        safe_prob = 0.0
        risk_prob = 0.0

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]  # [prob_class0, prob_class1]
            safe_prob = round(float(proba[0]) * 100, 2)
            risk_prob = round(float(proba[1]) * 100, 2)
        else:
            # If model doesn't support predict_proba
            safe_prob = 100.0 if pred == 0 else 0.0
            risk_prob = 100.0 if pred == 1 else 0.0

        confidence = round(max(safe_prob, risk_prob), 2)

        # Save to SQLite (history)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO history (
                age, income, loan_amount, credit_score, dti,
                education, employment_type, prediction
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (age, income, loan_amt, credit, dti, edu, emp, pred))
        conn.commit()
        conn.close()

        status = "Loan Approved ✅" if pred == 0 else "Loan Rejected ❌"

        return render_template(
            "result.html",
            result=status,
            prediction=pred,
            safe_prob=safe_prob,
            risk_prob=risk_prob,
            confidence=confidence,
            # Optional: show user input again on result page
            age=age,
            income=income,
            loan_amount=loan_amt,
            credit_score=credit,
            dti_ratio=dti,
            education=edu,
            employment_type=emp
        )

    except Exception as e:
        return render_template("index.html", error=str(e))

# ✅ Reviews
@app.route("/reviews", methods=["GET", "POST"])
def reviews():
    if not template_exists("reviews.html"):
        return redirect(url_for("dashboard"))

    page_error = None

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        if request.method == "POST":
            name = (request.form.get("name") or "").strip()
            tag = (request.form.get("tag") or "").strip()
            message = (request.form.get("message") or "").strip()

            try:
                rating = int(request.form.get("rating", 5))
            except:
                rating = 5

            rating = max(1, min(5, rating))

            cur.execute("""
                INSERT INTO reviews (name, tag, rating, message)
                VALUES (?, ?, ?, ?)
            """, (name, tag, rating, message))

            conn.commit()
            conn.close()
            return redirect(url_for("reviews"))

        cur.execute("SELECT COUNT(*), AVG(rating) FROM reviews")
        total_reviews, avg_rating = cur.fetchone()
        avg_rating = round(avg_rating, 1) if avg_rating is not None else 0

        df = pd.read_sql_query("""
            SELECT
                name,
                tag,
                rating,
                message,
                strftime('%d-%m-%Y', created_at) AS date
            FROM reviews
            ORDER BY id DESC
            LIMIT 30
        """, conn)

        conn.close()

        return render_template(
            "reviews.html",
            reviews=df.to_dict(orient="records"),
            total_reviews=total_reviews,
            avg_rating=avg_rating,
            page_error=page_error
        )

    except Exception as e:
        page_error = str(e)
        return render_template(
            "reviews.html",
            reviews=[],
            total_reviews=0,
            avg_rating=0,
            page_error=page_error
        )

@app.route("/about")
def about():
    if template_exists("about.html"):
        return render_template("about.html")
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
