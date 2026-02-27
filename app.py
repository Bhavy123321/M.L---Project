from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import pandas as pd

app = Flask(__name__)

# -----------------------------
# Load model (Render-safe path)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")
model = joblib.load(MODEL_PATH)

# If you use dropdowns in your HTML, keep these
EDUCATION_OPTIONS = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT_OPTIONS = ["Full-time", "Part-time", "Self-employed", "Unemployed"]


# -----------------------------
# Home -> Dashboard
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("dashboard.html")


# Optional direct dashboard route too
@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


# Prediction form page (if your form is on index.html)
@app.route("/predictor", methods=["GET"])
def predictor():
    return render_template(
        "index.html",
        education_options=EDUCATION_OPTIONS,
        employment_options=EMPLOYMENT_OPTIONS,
    )


# -----------------------------
# Predict
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Match your form field names exactly
        age = float(request.form.get("Age", 0))
        income = float(request.form.get("Income", 0))
        loan_amount = float(request.form.get("LoanAmount", 0))
        credit_score = float(request.form.get("CreditScore", 0))
        dti = float(request.form.get("DTIRatio", 0))
        education = request.form.get("Education", "")
        employment = request.form.get("EmploymentType", "")

        data = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "DTIRatio": dti,
            "Education": education,
            "EmploymentType": employment,
        }])

        pred = int(model.predict(data)[0])

        # Adjust label if your model uses 0/1 differently
        result = "Loan Approved ✅" if pred == 0 else "Loan Rejected ❌"

        return render_template("result.html", result=result)

    except Exception as e:
        # Send back to predictor page with an error
        return render_template(
            "index.html",
            education_options=EDUCATION_OPTIONS,
            employment_options=EMPLOYMENT_OPTIONS,
            error=str(e),
        )


# -----------------------------
# IMPORTANT: Never show "Not Found"
# Redirect any unknown route to dashboard
# -----------------------------
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("home"))


# -----------------------------
# Local run (Render uses gunicorn, this is for local testing)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
