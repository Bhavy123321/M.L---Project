from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
import joblib
import pandas as pd

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")          # history storage
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")    # your model file


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("MODEL LOAD ERROR:", e)


# -------------------------------------------------
# DB: only for prediction history (NO USERS/LOGIN)
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
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

init_db()


def template_exists(name: str) -> bool:
    return os.path.exists(os.path.join(BASE_DIR, "templates", name))


# -------------------------------------------------
# ROUTES
# -------------------------------------------------

# ✅ Home always opens dashboard
@app.route("/", methods=["GET"])
def dashboard():
    # Read history if exists
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()

    total = len(df)
    approved = len(df[df["prediction"] == 0])  # if your model uses 0=Approved
    rejected = len(df[df["prediction"] == 1])  # and 1=Rejected
    history = df.to_dict(orient="records")

    if template_exists("dashboard.html"):
        return render_template(
            "dashboard.html",
            total=total,
            approved=approved,
            rejected=rejected,
            history=history
        )

    # Fallback if dashboard.html missing
    return f"Dashboard is running ✅ Total predictions: {total}"


# ✅ Prediction page (GET) + Prediction logic (POST)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # Your repo has index.html for the form
        if template_exists("index.html"):
            return render_template("index.html")
        if template_exists("predict.html"):
            return render_template("predict.html")
        return "Prediction form template not found (index.html / predict.html)."

    # POST -> run prediction
    if model is None:
        return "Model not loaded. Check loan_model.pkl path and requirements."

    try:
        # Match your form field names (edit these only if your HTML uses different names)
        age = float(request.form.get("Age", 0))
        income = float(request.form.get("Income", 0))
        loan_amount = float(request.form.get("LoanAmount", 0))
        credit_score = float(request.form.get("CreditScore", 0))
        dti = float(request.form.get("DTIRatio", 0))
        education = request.form.get("Education", "")
        employment_type = request.form.get("EmploymentType", "")

        # DataFrame for model
        X = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "DTIRatio": dti,
            "Education": education,
            "EmploymentType": employment_type
        }])

        pred = int(model.predict(X)[0])

        # If your model uses reverse labels, swap these two lines:
        result = "Loan Approved ✅" if pred == 0 else "Loan Rejected ❌"

        # Save history
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO history (age, income, loan_amount, credit_score, dti, education, employment_type, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (age, income, loan_amount, credit_score, dti, education, employment_type, pred))
        conn.commit()
        conn.close()

        # Render result page
        if template_exists("result.html"):
            return render_template("result.html", result=result, prediction=pred)
        return result

    except Exception as e:
        return f"Prediction Error: {str(e)}"


# Optional pages (only if templates exist)
@app.route("/about")
def about():
    if template_exists("about.html"):
        return render_template("about.html")
    return redirect(url_for("dashboard"))

@app.route("/contact")
def contact():
    if template_exists("contact.html"):
        return render_template("contact.html")
    return redirect(url_for("dashboard"))

@app.route("/reviews")
def reviews():
    if template_exists("reviews.html"):
        return render_template("reviews.html")
    return redirect(url_for("dashboard"))


# ✅ No login/logout. If something hits /login or /logout, send to dashboard.
@app.route("/login")
@app.route("/logout")
def no_auth_routes():
    return redirect(url_for("dashboard"))


# ✅ IMPORTANT: Never show Not Found — redirect unknown routes to dashboard
@app.errorhandler(404)
def handle_404(e):
    return redirect(url_for("dashboard"))


# Local run (Render uses gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
