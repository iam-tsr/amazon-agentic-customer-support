"""
Intent classifier wrapper — called as a subprocess by main.ts.

Usage:
    python3.11 src/intent_classify.py <text>

Loads the sklearn TF-IDF + LogisticRegression pipeline from
src/model/tweet_classify.joblib and prints the predicted label to stdout.

Labels: complaint | other | positive | question
"""

import sys
import warnings

warnings.filterwarnings("ignore")   # suppress sklearn version mismatch noise

import joblib  # noqa: E402

MODEL_PATH = "src/model/tweet_classify.joblib"

_model = joblib.load(MODEL_PATH)

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print("other")
        sys.exit(0)
    label = _model.predict([text])[0]
    print(label)
