import os
import re
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_model" / "imdb_sentiment_lstm.keras"
TOKENIZER_PATH = BASE_DIR / "saved_model" / "tokenizer.pkl"
MAX_LENGTH = 100

model = None
tokenizer = None
load_error = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Put imdb_sentiment_lstm.keras "
            "inside the saved_model folder."
        )
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. Put tokenizer.pkl "
            "inside the saved_model folder."
        )

    model = load_model(MODEL_PATH)
    # Only load pickle files that you created and trust.
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
except Exception as exc:
    load_error = str(exc)
    app.logger.exception("Could not load the sentiment model or tokenizer.")


def clean_text(text):
    """Apply the same text cleaning used when preparing the training data."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_review(review):
    """Return the model score and the corresponding binary sentiment label."""
    cleaned = clean_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="post"
    )

    positive_score = float(model.predict(padded, verbose=0)[0][0])
    positive_score = min(max(positive_score, 0.0), 1.0)
    negative_score = 1.0 - positive_score

    if positive_score >= 0.5:
        label = "Positive"
        confidence = positive_score
        explanation = (
            "The model's output is at or above its 0.50 decision threshold, "
            "so it assigns this review to the positive class."
        )
    else:
        label = "Negative"
        confidence = negative_score
        explanation = (
            "The model's output is below its 0.50 decision threshold, "
            "so it assigns this review to the negative class."
        )

    return {
        "label": label,
        "positive_pct": round(positive_score * 100, 1),
        "negative_pct": round(negative_score * 100, 1),
        "confidence_pct": round(confidence * 100, 1),
        "explanation": explanation,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    review = ""
    result = None
    error = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if load_error:
            error = (
                "The model could not be loaded. Check that both saved files "
                "are in the saved_model folder, then restart the app."
            )
        elif not review:
            error = "Please enter a movie review before analyzing it."
        elif len(review) > 10000:
            error = "Please keep the review under 10,000 characters."
        else:
            try:
                result = predict_review(review)
            except Exception:
                app.logger.exception("Prediction failed.")
                error = (
                    "Sorry, the review could not be analyzed. Check the model "
                    "and tokenizer files, then try again."
                )

    return render_template(
        "index.html",
        review=review,
        result=result,
        error=error,
        model_ready=(model is not None and tokenizer is not None and not load_error),
        load_error=load_error,
    )


if __name__ == "__main__":
    # Run locally. Debug is disabled by default.
    app.run(host="127.0.0.1", port=5000, debug=False)
