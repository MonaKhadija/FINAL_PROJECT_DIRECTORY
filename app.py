import re
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    raw_text = data["text"]
    cleaned = clean_text(raw_text)

    # Vectorize and Predict
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized).max()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "text": raw_text,
        "sentiment": sentiment,
        "confidence": float(probability)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)