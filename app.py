from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
try:
    model = joblib.load("risk_model.pkl")
    print("✅ ML model loaded successfully")
except Exception as e:
    print("❌ ML model load failed:", e)
    model = None
@app.route("/")
def home():
    return render_template("page7.html") 
def score_tasks(answers):
    scores = {}
    def word_count_score(text):
        wc = len(text.split())
        if wc > 20:
            return 10
        elif wc > 10:
            return 7
        elif wc > 5:
            return 4
        else:
            return 2
    scores["Picture Description Task"] = word_count_score(
        answers.get("Picture Description Task", "")
    )
    scores["Story Reading and Word Repetition Detection"] = (
        10 if "umbrella" in answers.get(
            "Story Reading and Word Repetition Detection", "").lower() else 5
    )
    scores["Word Fluency Test"] = (
        10 if answers.get("Word Fluency Test", "") else 5
    )
    scores["Visual Memory"] = (
        10 if answers.get("Visual Memory", "").upper() == "B" else 5
    )
    scores["Problem-Solving and Conditional Logic"] = (
        10 if "5" in answers.get(
            "Problem-Solving and Conditional Logic", "") else 5
    )
    scores["Object-Location Recall"] = word_count_score(
        answers.get("Object-Location Recall", "")
    )
    scores["Sentence Repetition Task"] = (
        10 if "quick brown fox" in answers.get(
            "Sentence Repetition Task", "").lower() else 5
    )
    scores["Critical Thinking Scenario"] = (
        10 if any(w in answers.get(
            "Critical Thinking Scenario", "").lower()
                for w in ["call", "run", "safe", "help"]) else 5
    )
    scores["Delayed Recall Task"] = (
        10 if "umbrella" in answers.get(
            "Delayed Recall Task", "").lower() else 5
    )
    scores["Pattern Continuation and Logical Reasoning Task"] = (
        10 if answers.get(
            "Pattern Continuation and Logical Reasoning Task", "") else 5
    )
    features = [
        scores["Picture Description Task"],
        scores["Story Reading and Word Repetition Detection"],
        scores["Word Fluency Test"],
        scores["Visual Memory"],
        scores["Problem-Solving and Conditional Logic"],
        scores["Object-Location Recall"],
        scores["Sentence Repetition Task"],
        scores["Critical Thinking Scenario"],
        scores["Delayed Recall Task"],
        scores["Pattern Continuation and Logical Reasoning Task"],
    ]

    return scores, features


@app.route("/calculate_risk", methods=["POST"])
def calculate_risk():
    print("🔥 calculate_risk API HIT 🔥")

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data"}), 400

        task_scores, features = score_tasks(data)

        if model is None:
            return jsonify({"error": "ML model not loaded"}), 500

        prediction = model.predict([features])[0]

        if hasattr(model, "predict_proba"):
            risk_percent = model.predict_proba([features])[0][prediction] * 100
        else:
            risk_percent = (sum(features) / (len(features) * 10)) * 100

        if prediction == 0:
            risk_level = "Low"
        elif prediction == 1:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        print("Features:", features)
        print("Prediction:", prediction)

        return jsonify({
            "risk_level": risk_level,
            "risk_percent": round(risk_percent, 2),
            "task_scores": task_scores
        })

    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
