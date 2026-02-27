from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# --------------------------------------------------
# Load Model (Render-safe path)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")
model = joblib.load(MODEL_PATH)

# Dropdown values
EDUCATION_OPTIONS = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT_OPTIONS = ["Full-time", "Part-time", "Self-employed", "Unemployed"]


# --------------------------------------------------
# HOME → Dashboard directly
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("dashboard.html")


# --------------------------------------------------
# Prediction Form Page
# --------------------------------------------------
@app.route("/predictor")
def predictor():
    return render_template(
        "index.html",
        education_options=EDUCATION_OPTIONS,
        employment_options=EMPLOYMENT_OPTIONS,
    )


# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["Age"])
        income = float(request.form["Income"])
        loan_amount = float(request.form["LoanAmount"])
        credit_score = float(request.form["CreditScore"])
        dti = float(request.form["DTIRatio"])
        education = request.form["Education"]
        employment = request.form["EmploymentType"]

        # Create dataframe
        data = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "DTIRatio": dti,
            "Education": education,
            "EmploymentType": employment
        }])

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Loan Rejected ❌"
        else:
            result = "Loan Approved ✅"

        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error: {str(e)}"


# --------------------------------------------------
# Run App (important for Render)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
