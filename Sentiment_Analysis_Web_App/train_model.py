"""Train and evaluate a 3-class TF-IDF + Logistic Regression sentiment model.

The bundled examples are for a runnable educational demo, not production use.
For a credible portfolio evaluation, replace them with a licensed real-world
dataset and use a held-out test set.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

examples = {
    "positive": [
        "I love this product", "This is amazing", "Excellent quality and fast delivery",
        "Very happy with my purchase", "The experience was wonderful",
        "Absolutely fantastic service", "It works perfectly", "Highly recommended",
        "Great value for money", "I am very satisfied", "The app is easy to use",
        "Beautiful design and great performance", "This was a good decision",
        "I really enjoyed it", "The support team was helpful", "The food was delicious",
        "The movie was entertaining", "The staff were friendly", "The battery lasts all day",
        "The instructions were clear", "I would buy this again", "Reliable and efficient",
        "The website is fast and intuitive", "A great experience overall",
        "The camera takes excellent photos", "The delivery arrived early",
        "The course was informative", "The update fixed my issue",
        "Comfortable fit and premium material", "The sound quality is excellent",
        "The service exceeded my expectations", "The interface looks clean",
        "This saved me a lot of time", "The book was engaging", "The room was spotless",
        "The setup was simple", "The product is worth the price",
        "I am impressed with the results", "Everything worked as expected",
        "The team was professional", "A useful and practical tool",
        "The plot was interesting", "The laptop is fast and reliable",
        "The order arrived safely", "The app makes my work easier",
        "The repair was completed quickly", "A pleasant and memorable visit",
        "The material feels durable", "The refund was processed quickly",
        "I had a fantastic day", "The teacher explained everything well",
        "This is a brilliant solution", "The quality is better than expected",
        "The new features are helpful", "The customer care was excellent",
        "It is simple, useful, and effective", "I am pleased with this purchase",
        "The performance is outstanding", "The room was quiet and comfortable",
        "The presentation was clear", "I strongly recommend this service"
    ],
    "negative": [
        "I hate this product", "This is terrible", "Very poor quality",
        "I am disappointed with this purchase", "The experience was awful",
        "Absolutely horrible service", "It stopped working", "Not recommended",
        "Waste of money", "I am very dissatisfied", "The app is confusing",
        "Bad design and poor performance", "This was a terrible decision",
        "I regret buying it", "The support team was unhelpful", "The food was disgusting",
        "The movie was boring", "The staff were rude", "The battery drains quickly",
        "The instructions were unclear", "I would not buy this again", "Unreliable and slow",
        "The website is broken and frustrating", "A disappointing experience overall",
        "The camera takes blurry photos", "The delivery arrived late",
        "The course was confusing", "The update created more problems",
        "Uncomfortable fit and cheap material", "The sound quality is awful",
        "The service was below expectations", "The interface is cluttered",
        "This wastes a lot of time", "The book was dull", "The room was dirty",
        "The setup was difficult", "The product is not worth the price",
        "I am unhappy with the results", "Nothing worked as expected",
        "The team was unprofessional", "A useless and impractical tool",
        "The plot made no sense", "The laptop is slow and unreliable",
        "The order arrived damaged", "The app makes my work harder",
        "The repair took too long", "A disappointing visit",
        "The material feels fragile", "The refund was delayed",
        "I had a terrible day", "The teacher explained nothing clearly",
        "This is a poor solution", "The quality is worse than expected",
        "The new features are annoying", "The customer care was awful",
        "It is difficult, frustrating, and ineffective", "I regret this purchase",
        "The performance is disappointing", "The room was noisy and uncomfortable",
        "The presentation was confusing", "I strongly dislike this service"
    ],
    "neutral": [
        "The product arrived yesterday", "The package contains two items",
        "The app has a settings page", "The meeting starts at nine",
        "The book has twelve chapters", "The phone is available in blue",
        "The order number is listed on the receipt", "The store opens at ten",
        "The screen measures six inches", "The update was released on Monday",
        "The report includes three sections", "The product is made of plastic",
        "The website has a search box", "The class lasts one hour",
        "The package weighs two kilograms", "The device has a power button",
        "The movie is two hours long", "The hotel is near the station",
        "The file is saved in the folder", "The form asks for an email address",
        "The laptop has sixteen gigabytes of memory", "The bus leaves at noon",
        "The product comes in three sizes", "The page displays a list of items",
        "The app supports English", "The office is on the second floor",
        "The order was placed on Tuesday", "The course has five modules",
        "The camera has a flash", "The document contains the project details",
        "The meeting is scheduled for tomorrow", "The store sells electronic items",
        "The screen brightness can be adjusted", "The form has four fields",
        "The service is available in this area", "The product code is printed on the box",
        "The book was published in 2022", "The website includes a contact page",
        "The file size is ten megabytes", "The train arrives at platform three",
        "The application has a login page", "The device connects using USB",
        "The table has four columns", "The video lasts five minutes",
        "The package was delivered this morning", "The product page lists its dimensions",
        "The class has thirty students", "The room has two windows",
        "The report was saved as a PDF", "The app was installed today",
        "The screen resolution is listed in settings", "The store is closed on Sunday",
        "The email includes an attachment", "The device has a rechargeable battery",
        "The ticket costs five hundred rupees", "The page has a footer",
        "The product is listed under electronics", "The form was submitted at noon",
        "The file contains a table", "The website was updated last week"
    ]
}

texts, labels = [], []
for label, sentences in examples.items():
    texts.extend(sentences)
    labels.extend([label] * len(sentences))

df = pd.DataFrame({"text": texts, "label": labels})
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                              ngram_range=(1, 2), sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced",
                                      random_state=42))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

joblib.dump(model, "sentiment_pipeline.joblib")
print("Model saved to sentiment_pipeline.joblib")
print(f"Demo held-out accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report (small synthetic demo dataset):")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix labels:", list(model.classes_))
print(confusion_matrix(y_test, predictions, labels=list(model.classes_)))
print("\nImportant: these metrics are only for the bundled educational examples.")
