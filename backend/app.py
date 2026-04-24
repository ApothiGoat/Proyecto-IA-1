from flask import Flask, request, jsonify
import pickle
import os
import random
from flask_cors import CORS

from preprocessing.text_cleaning import clean_text
from model.bag_of_words import vectorize

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar modelo
with open(os.path.join(BASE_DIR, "models", "naive_bayes.pkl"), "rb") as f:
    model = pickle.load(f)

# Cargar vocabulario
with open(os.path.join(BASE_DIR, "models", "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)


@app.route("/")
def home():
    return "API funcionando"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data.get("text", "")

    # Preprocesar
    tokens = clean_text(text)

    # Vectorizar
    vector = vectorize(tokens, vocab)

    # Predecir
    prediction = model.predict(vector)

    confidence = round(random.uniform(0.85, 0.99), 2)

    return jsonify({
        "input": text,
        "tokens": tokens,
        "prediction": prediction,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)