from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "loansense_secret_key"  # change if you want

DB_NAME = "database.db"


# ---------------------------
# DB helpers
# ---------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # Create DB file + table if not exists
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            income INTEGER NOT NULL,
            credit_score INTEGER NOT NULL,
            loan_amount INTEGER NOT NULL,
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# Initialize DB on startup
init_db()


# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    name = request.form.get("name", "").strip()
    income = request.form.get("income", "").strip()
    credit_score = request.form.get("credit_score", "").strip()
    loan_amount = request.form.get("loan_amount", "").strip()

    # Basic validation
    if not name or not income or not credit_score or not loan_amount:
        flash("Please fill all fields!", "error")
        return redirect(url_for("home"))

    try:
        income = int(income)
        credit_score = int(credit_score)
        loan_amount = int(loan_amount)
    except ValueError:
        flash("Income, Credit Score, and Loan Amount must be numbers!", "error")
        return redirect(url_for("home"))

    # Simple logic for result (you can replace with ML model later)
    # Example rule:
    # if credit_score >= 700 and income >= 25000 and loan_amount <= income * 10 => Approved
    if credit_score >= 700 and income >= 25000 and loan_amount <= income * 10:
        result = "Approved"
    else:
        result = "Rejected"

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO loan_applications (name, income, credit_score, loan_amount, result)
        VALUES (?, ?, ?, ?, ?)
    """, (name, income, credit_score, loan_amount, result))
    conn.commit()
    conn.close()

    flash("Data saved successfully ✅", "success")
    return redirect(url_for("applications"))


@app.route("/applications")
def applications():
    conn = get_db_connection()
    data = conn.execute("SELECT * FROM loan_applications ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("applications.html", data=data)


@app.route("/delete/<int:id>", methods=["POST"])
def delete(id):
    conn = get_db_connection()
    conn.execute("DELETE FROM loan_applications WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    flash("Deleted successfully 🗑️", "success")
    return redirect(url_for("applications"))


if __name__ == "__main__":
    app.run(debug=True)
