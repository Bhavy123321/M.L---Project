from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sqlite3
import joblib
import pandas as pd

# -------------------------------------------------
# PATHS / APP
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")

app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------------------------------
# GLOBAL TEMPLATE VARIABLES
# -------------------------------------------------
@app.context_processor
def inject_globals():
    return {
        "brand_name": "Loan Default",
        "social": {
            "linkedin": "#", "github": "#", "instagram": "#", "twitter": "#",
        },
    }

# -------------------------------------------------
# DATABASE INITIALIZATION
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL, income REAL, loan_amount REAL,
            credit_score REAL, dti REAL, education TEXT,
            employment_type TEXT, prediction INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------------------------------------
# MODEL LOADER
# -------------------------------------------------
_model = None
def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
    return _model

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.route("/", methods=["GET"])
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    # Using column aliases to match your HTML r.Age, r.Income, etc.
    df = pd.read_sql_query("""
        SELECT 
            created_at, 
            age AS Age, 
            income AS Income, 
            loan_amount AS LoanAmount, 
            credit_score AS CreditScore, 
            dti AS DTIRatio, 
            CASE WHEN prediction = 0 THEN 'Approved' ELSE 'Rejected' END as result,
            prediction
        FROM history ORDER BY id DESC
    """, conn)
    
    # Trend data: count of predictions per date
    trend_df = pd.read_sql_query("""
        SELECT DATE(created_at) as date, COUNT(*) as count 
        FROM history 
        GROUP BY DATE(created_at) 
        ORDER BY date ASC LIMIT 10
    """, conn)
    conn.close()

    total = len(df)
    approved = int((df["prediction"] == 0).sum()) if total else 0
    rejected = int((df["prediction"] == 1).sum()) if total else 0
    
    # Matches {% for r in recent %}
    recent = df.to_dict(orient="records")

    # Matches {{ trend_labels|tojson }} and {{ trend_counts|tojson }}
    trend_labels = trend_df['date'].tolist() if not trend_df.empty else []
    trend_counts = trend_df['count'].tolist() if not trend_df.empty else []

    return render_template(
        "dashboard.html",
        total=total,
        approved=approved,
        rejected=rejected,
        recent=recent,
        trend_labels=trend_labels,
        trend_counts=trend_counts
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")

    try:
        model = get_model()
        # Form inputs
        age = float(request.form.get("Age", 0))
        income = float(request.form.get("Income", 0))
        loan_amount = float(request.form.get("LoanAmount", 0))
        credit_score = float(request.form.get("CreditScore", 0))
        dti = float(request.form.get("DTIRatio", 0))
        edu = request.form.get("Education", "")
        emp = request.form.get("EmploymentType", "")

        # DataFrame for model
        X = pd.DataFrame([{
            "Age": age, "Income": income, "LoanAmount": loan_amount,
            "CreditScore": credit_score, "DTIRatio": dti,
            "Education": edu, "EmploymentType": emp
        }])

        pred = int(model.predict(X)[0])
        
        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO history (age, income, loan_amount, credit_score, dti, education, employment_type, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (age, income, loan_amount, credit_score, dti, edu, emp, pred))
        conn.commit()
        conn.close()

        result_text = "Loan Approved ✅" if pred == 0 else "Loan Rejected ❌"
        return render_template("result.html", result=result_text, prediction=pred)

    except Exception as e:
        return render_template("index.html", error=str(e))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/reviews")
def reviews():
    return render_template("reviews.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
