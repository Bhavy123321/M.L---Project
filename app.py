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
            "linkedin": "#",
            "github": "#",
            "instagram": "#",
            "twitter": "#",
        },
    }

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def log(msg: str):
    print(f"[APP] {msg}", flush=True)

def template_exists(name: str) -> bool:
    return os.path.exists(os.path.join(BASE_DIR, "templates", name))

def init_db():
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()
    log("DB ready ✅")

init_db()

_model = None
def get_model():
    global _model
    if _model is None:
        log("Loading model...")
        _model = joblib.load(MODEL_PATH)
        log("Model loaded ✅")
    return _model

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/", methods=["GET"])
def dashboard():
    log("GET /")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()

    total = len(df)
    approved = int((df["prediction"] == 0).sum()) if total else 0
    rejected = int((df["prediction"] == 1).sum()) if total else 0
    history = df.to_dict(orient="records") if total else []
    
    # FIX: Ensure these variables are defined for the template
    trend_labels = df["created_at"].tolist() if total else []
    trend_data = df["prediction"].tolist() if total else []

    if template_exists("dashboard.html"):
        return render_template(
            "dashboard.html",
            total=total,
            approved=approved,
            rejected=rejected,
            history=history,
            trend_labels=trend_labels,
            trend_data=trend_data
        )
    return f"Dashboard running ✅ Total predictions: {total}"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html") if template_exists("index.html") else "index.html missing", 404

    try:
        model = get_model()
        data = {
            "Age": float(request.form.get("Age", 0)),
            "Income": float(request.form.get("Income", 0)),
            "LoanAmount": float(request.form.get("LoanAmount", 0)),
            "CreditScore": float(request.form.get("CreditScore", 0)),
            "DTIRatio": float(request.form.get("DTIRatio", 0)),
            "Education": str(request.form.get("Education", "")).strip(),
            "EmploymentType": str(request.form.get("EmploymentType", "")).strip(),
        }
        X = pd.DataFrame([data])
        pred = int(model.predict(X)[0])
        
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""INSERT INTO history (age, income, loan_amount, credit_score, dti, education, employment_type, prediction) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                    (data["Age"], data["Income"], data["LoanAmount"], data["CreditScore"], 
                     data["DTIRatio"], data["Education"], data["EmploymentType"], pred))
        conn.commit()
        conn.close()

        return render_template("result.html", result="Loan Approved ✅" if pred == 0 else "Loan Rejected ❌", prediction=pred)
    except Exception as e:
        log(f"Predict error: {e}")
        return str(e), 400

@app.route("/about")
def about(): return render_template("about.html") if template_exists("about.html") else redirect(url_for("dashboard"))

@app.route("/reviews")
def reviews(): return render_template("reviews.html") if template_exists("reviews.html") else redirect(url_for("dashboard"))

@app.errorhandler(404)
def handle_404(e): return redirect(url_for("dashboard"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
