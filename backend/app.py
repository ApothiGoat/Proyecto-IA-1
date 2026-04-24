from flask import Flask, request, jsonify
import pickle
import os

from preprocessing.text_cleaning import clean_text
from model.bag_of_words import vectorize

app = Flask(__name__)

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

    return jsonify({
        "input": text,
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)