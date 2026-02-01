from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)
DB_NAME = "database.db"


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            income INTEGER NOT NULL,
            credit_score INTEGER NOT NULL,
            loan_amount INTEGER NOT NULL,
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


init_db()


def predict_logic(income: int, credit_score: int, loan_amount: int) -> str:
    # Replace with your ML model prediction if you have one.
    if credit_score >= 700 and income >= 25000 and loan_amount <= income * 10:
        return "Approved"
    return "Rejected"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    history = []

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        income = request.form.get("income", "").strip()
        credit_score = request.form.get("credit_score", "").strip()
        loan_amount = request.form.get("loan_amount", "").strip()

        # basic validation
        if not name or not income or not credit_score or not loan_amount:
            result = "Please fill all fields!"
        else:
            try:
                income = int(income)
                credit_score = int(credit_score)
                loan_amount = int(loan_amount)
            except ValueError:
                result = "Income / Credit Score / Loan Amount must be numbers!"
            else:
                result = predict_logic(income, credit_score, loan_amount)

                # Save current record
                conn = get_db()
                conn.execute("""
                    INSERT INTO loan_applications (name, income, credit_score, loan_amount, result)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, income, credit_score, loan_amount, result))
                conn.commit()

                # Fetch all history
                history = conn.execute("""
                    SELECT * FROM loan_applications ORDER BY id DESC
                """).fetchall()
                conn.close()

    return render_template("index.html", result=result, history=history)


if __name__ == "__main__":
    app.run(debug=True)
