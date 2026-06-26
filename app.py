
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline

app = Flask(__name__)

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sentiment.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Database Model
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    score = db.Column(db.Float, nullable=False)

# Load AI Model
print("Loading model...")
classifier = pipeline("sentiment-analysis")
print("Model loaded!")

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        text = request.form.get("text")

        if text:

            prediction = classifier(text)

            sentiment = prediction[0]["label"]
            score = round(prediction[0]["score"] * 100, 2)

            # Save in database
            new_record = Analysis(
                text=text,
                sentiment=sentiment,
                score=score
            )

            db.session.add(new_record)
            db.session.commit()

            result = {
                "text": text,
                "sentiment": sentiment,
                "score": score
            }

    return render_template(
        "index.html",
        result=result
    )

# REST API Endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json()

    if not data or "text" not in data:
        return {"error": "Please provide text"}, 400

    prediction = classifier(data["text"])

    return {
        "text": data["text"],
        "sentiment": prediction[0]["label"],
        "score": round(prediction[0]["score"] * 100, 2)
    }

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)