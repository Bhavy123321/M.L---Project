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
    cur.execute(
        """
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
        """
    )
    conn.commit()
    conn.close()
    log("DB ready ✅")


init_db()

# lazy-load model
_model = None


def get_model():
    global _model
    if _model is None:
        log("Loading model...")
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            log("Model loaded ✅")
        else:
            log("ERROR: Model file not found!")
    return _model


# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/health")
def health():
    return jsonify(
        status="ok",
        has_model_file=os.path.exists(MODEL_PATH),
        has_db_file=os.path.exists(DB_PATH),
    )


@app.route("/", methods=["GET"])
def dashboard():
    log("GET /")

    conn = sqlite3.connect(DB_PATH)
    # Load all history for the table
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    
    # Load trend data (last 10 entries) for the chart
    trend_df = pd.read_sql_query("SELECT created_at, prediction FROM history ORDER BY id ASC LIMIT 10", conn)
    conn.close()

    total = len(df)
    approved = int((df["prediction"] == 0).sum()) if total else 0
    rejected = int((df["prediction"] == 1).sum()) if total else 0
    history = df.to_dict(orient="records") if total else []

    # Prepare data for the JavaScript charts in dashboard.html
    # This prevents the "Undefined is not JSON serializable" error
    trend_labels = trend_df['created_at'].tolist() if not trend_df.empty else []
    trend_data = trend_df['prediction'].tolist() if not trend_df.empty else []

    if template_exists("dashboard.html"):
        return render_template(
            "dashboard.html",
            total=total,
            approved=approved,
            rejected=rejected,
            history=history,
            trend_labels=trend_labels,  # Required by line 124 of your template
            trend_data=trend_data       # Likely required by your chart JS
        )

    return f"Dashboard running ✅ Total predictions: {total}"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    log(f"{request.method} /predict")

    if request.method == "GET":
        if template_exists("index.html"):
            return render_template("index.html")
        return "index.html not found.", 404

    try:
        model = get_model()
        if model is None:
            raise Exception("Machine Learning model file is missing on the server.")

        age = float(request.form.get("Age", 0))
        income = float(request.form.get("Income", 0))
        loan_amount = float(request.form.get("LoanAmount", 0))
        credit_score = float(request.form.get("CreditScore", 0))
        dti = float(request.form.get("DTIRatio", 0))
        education = (request.form.get("Education") or "").strip()
        employment_type = (request.form.get("EmploymentType") or "").strip()

        X = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "DTIRatio": dti,
            "Education": education,
            "EmploymentType": employment_type,
        }])

        pred = int(model.predict(X)[0])
        result = "Loan Approved ✅" if pred == 0 else "Loan Rejected ❌"

        # save to DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO history (age, income, loan_amount, credit_score, dti, education, employment_type, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (age, income, loan_amount, credit_score, dti, education, employment_type, pred),
        )
        conn.commit()
        conn.close()

        if template_exists("result.html"):
            return render_template("result.html", result=result, prediction=pred)

        return result

    except Exception as e:
        log(f"Predict error: {e}")
        if template_exists("index.html"):
            return render_template("index.html", error=str(e), form=request.form), 400
        return f"Prediction Error: {str(e)}", 500


@app.route("/about")
def about():
    return render_template("about.html") if template_exists("about.html") else redirect(url_for("dashboard"))


@app.route("/reviews")
def reviews():
    return render_template("reviews.html") if template_exists("reviews.html") else redirect(url_for("dashboard"))


@app.route("/login")
@app.route("/logout")
def no_auth_routes():
    return redirect(url_for("dashboard"))


@app.errorhandler(404)
def handle_404(e):
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
