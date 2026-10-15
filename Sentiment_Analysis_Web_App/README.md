# SentimentIQ — Advanced Sentiment Analysis Web App

A portfolio-oriented educational prototype with:
- Three-way sentiment classification: positive, negative, neutral
- TF-IDF text features with word and bigram signals
- Logistic Regression classifier
- Class probability estimates and top-score display
- Flask web UI with responsive styling
- Recent predictions stored temporarily in the browser session
- Batch CSV prediction and downloadable output
- Train/test split and classification report during training

## Run on Windows (VS Code)

Open the extracted project folder in VS Code. Open Terminal in the folder that contains `app.py` and `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Keep the terminal open while using the app.

If `requirements.txt` cannot be found, run `dir` and confirm that the terminal is in the project root. Do not run commands from a parent or nested folder.

## Batch CSV
Open the Batch CSV tab and upload `data/sample_reviews.csv`. The CSV must contain a column named `text`. Results can be downloaded as `sentiment_predictions.csv`.

## Interview explanation
“I built a sentiment analysis web application using Python, Scikit-learn and Flask. TF-IDF converts text into numerical features, and Logistic Regression predicts positive, negative or neutral sentiment. The web UI displays the predicted class and model probability estimates. I also added batch CSV inference so multiple reviews can be analyzed at once. The current dataset is a small synthetic educational dataset, so I would validate the system on a larger, licensed real-world dataset before making accuracy claims.”

## Limitations and responsible reporting
The included examples are synthetic and intentionally small, so results are for demonstration only. Do not claim production-grade performance. A real portfolio version should use a licensed dataset, deduplicate examples, split data before model selection, evaluate macro-F1 and per-class precision/recall on a held-out set, test domain shift, and document known failure cases. Probability scores are model estimates and should not be described as guaranteed certainty.

Before public deployment, replace `app.secret_key`, disable Flask debug mode, validate uploads more strictly, and use production hosting.
