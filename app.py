from flask import Flask, render_template, request
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)
DB_NAME = "database.db"


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            form_json TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ✅ Put your existing ML predict logic here (keep your old one if you already have)
def predict_loan(form_dict: dict) -> str:
    """
    Replace this with your existing model prediction logic.
    For now, it does a simple demo rule if fields exist.
    """
    # If your form has credit_score & income fields, this will work.
    # Otherwise it will just return "Approved" as default.
    try:
        credit_score = int(form_dict.get("credit_score", 700))
        income = int(form_dict.get("income", 25000))
    except:
        credit_score, income = 700, 25000

    if credit_score >= 700 and income >= 25000:
        return "Approved"
    return "Rejected"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    history = []

    if request.method == "POST":
        # ✅ Take whatever your existing UI form sends
        form_data = request.form.to_dict(flat=True)

        # ✅ Predict (use your old logic here)
        result = predict_loan(form_data)

        # ✅ Save current entry
        conn = get_db()
        conn.execute(
            "INSERT INTO loan_history (form_json, result, created_at) VALUES (?, ?, ?)",
            (json.dumps(form_data), result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

        # ✅ Fetch all history (including current)
        rows = conn.execute("SELECT * FROM loan_history ORDER BY id DESC").fetchall()
        conn.close()

        # Convert JSON back to dict for template
        history = [
            {
                "id": r["id"],
                "data": json.loads(r["form_json"]),
                "result": r["result"],
                "created_at": r["created_at"]
            }
            for r in rows
        ]

    # ✅ IMPORTANT: keep your SAME template name here.
    # If your first code used index.html, keep it.
    # If it used home.html / predict.html, change this line only.
    return render_template("index.html", result=result, history=history)


if __name__ == "__main__":
    app.run(debug=True)
