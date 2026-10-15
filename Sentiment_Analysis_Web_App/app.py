from flask import Flask, render_template, request, send_file, session
from pathlib import Path
from io import StringIO, BytesIO
import csv
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret-before-deployment"
MODEL_FILE = Path("sentiment_pipeline.joblib")
model = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

def predict_text(text):
    label = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    scores = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, probabilities)}
    confidence = max(scores.values()) if scores else 0.0
    return str(label), confidence, scores

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    text = ""
    if request.method == "POST":
        text = request.form.get("review", "").strip()
        if not text:
            error = "Please enter a review or sentence."
        elif len(text) > 5000:
            error = "Please keep the text under 5,000 characters."
        elif model is None:
            error = "Model not found. Run `python train_model.py` first."
        else:
            label, confidence, scores = predict_text(text)
            result = {"label": label, "confidence": confidence, "scores": scores}
            history = session.get("history", [])
            history.insert(0, {
                "text": text[:120],
                "label": label,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            session["history"] = history[:5]
    return render_template("index.html", result=result, error=error,
                           review=text, history=session.get("history", []),
                           model_ready=model is not None)

@app.route("/batch", methods=["GET", "POST"])
def batch():
    error = None
    rows = []
    if request.method == "POST":
        upload = request.files.get("file")
        if not upload or not upload.filename.lower().endswith(".csv"):
            error = "Upload a CSV file with a column named 'text'."
        elif model is None:
            error = "Model not found. Run `python train_model.py` first."
        else:
            try:
                df = pd.read_csv(upload)
                if "text" not in df.columns:
                    error = "CSV must contain a column named 'text'."
                elif len(df) > 1000:
                    error = "Please upload at most 1,000 rows at a time."
                else:
                    texts = df["text"].fillna("").astype(str).tolist()
                    preds = model.predict(texts)
                    probs = model.predict_proba(texts)
                    classes = list(model.classes_)
                    output = pd.DataFrame({"text": texts, "sentiment": preds})
                    output["confidence_percent"] = (probs.max(axis=1) * 100).round(1)
                    for idx, cls in enumerate(classes):
                        output[f"probability_{cls}_percent"] = (probs[:, idx] * 100).round(1)
                    session["batch_csv"] = output.to_csv(index=False)
                    rows = output.head(20).to_dict(orient="records")
            except Exception as exc:
                error = f"Could not read this CSV: {exc}"
    return render_template("batch.html", error=error, rows=rows)

@app.route("/download-results")
def download_results():
    content = session.get("batch_csv")
    if not content:
        return "No batch results available. Upload a CSV first.", 400
    return send_file(BytesIO(content.encode("utf-8")), mimetype="text/csv",
                     as_attachment=True, download_name="sentiment_predictions.csv")

if __name__ == "__main__":
    app.run(debug=True)
