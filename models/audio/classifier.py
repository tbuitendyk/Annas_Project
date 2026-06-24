"""
models/audio/classifier.py — Profanity text classifier.

Two-layer system:
  Layer 1: word_lookup.scan() — fast, deterministic, JSON-driven
  Layer 2: trained sklearn classifier — context-aware, improves with labeling

The ML layer lets the model learn context — e.g. "ass" in "assess" is clean,
or "Jesus" as a character name vs. an exclamation.
"""

import json
from pathlib import Path
from loguru import logger

from models.audio.word_lookup import scan, highest_severity, rebuild

TRAINING_DATA_PATH = Path("models/audio/training_data.jsonl")
MODEL_PATH         = Path("models/audio/classifier.pkl")


class ProfanityClassifier:
    """
    classify(text) → (severity, hits)
    severity: "clean" | "mild" | "moderate" | "severe"
    hits: [(word, severity), ...]
    """

    def __init__(self):
        self._ml_model    = None
        self._vectorizer  = None
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                import pickle
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self._vectorizer = data["vectorizer"]
                self._ml_model   = data["model"]
                logger.info(f"Classifier: loaded trained model from {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Classifier: could not load model: {e}")

    def classify(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        if not text or not text.strip():
            return "clean", []

        # Layer 1: word lookup
        hits = scan(text)
        word_list_severity = highest_severity(hits)

        # Layer 2: ML override (if trained)
        ml_severity = None
        if self._ml_model is not None:
            try:
                features   = self._vectorizer.transform([text.lower()])
                ml_severity = self._ml_model.predict(features)[0]
            except Exception as e:
                logger.warning(f"ML classifier error: {e}")

        final = ml_severity if ml_severity is not None else word_list_severity
        return final, hits

    def is_flagged(self, text: str) -> bool:
        severity, _ = self.classify(text)
        return severity not in ("clean", None)

    def reload(self):
        """Reload ML model and rebuild word lookup after edits."""
        self._ml_model   = None
        self._vectorizer = None
        self._load_model()
        rebuild()
        logger.info("Classifier: reloaded")


def train(training_data_path: Path = TRAINING_DATA_PATH,
          model_path: Path = MODEL_PATH) -> dict:
    """Train the ML classifier from labeled examples."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score
    import pickle
    import numpy as np

    if not training_data_path.exists():
        return {"error": "no training data"}

    texts, labels = [], []
    with open(training_data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                texts.append(ex["text"].lower())
                labels.append(ex["label"])
            except Exception as e:
                logger.warning(f"Skipping malformed example: {e}")

    if len(texts) < 10:
        return {"error": f"only {len(texts)} examples, need 10+"}

    logger.info(f"Training on {len(texts)} examples...")

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1,
                                 analyzer="word", sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X, labels)

    scores   = cross_val_score(model, X, labels,
                               cv=min(5, len(texts) // 4), scoring="accuracy")
    accuracy = float(np.mean(scores))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "model": model}, f)

    from collections import Counter
    stats = {
        "examples": len(texts),
        "accuracy": round(accuracy, 3),
        "label_distribution": dict(Counter(labels)),
        "model_path": str(model_path),
    }
    logger.info(f"Training complete: {stats}")
    return stats


def add_example(text: str, label: str,
                path: Path = TRAINING_DATA_PATH) -> None:
    assert label in ("clean", "mild", "moderate", "severe"), \
        f"Invalid label: {label}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"text": text, "label": label}) + "\n")
    logger.info(f"Added training example: [{label}] {text!r}")
