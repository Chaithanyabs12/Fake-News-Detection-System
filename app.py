from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Assign labels
fake_df["label"] = 0
true_df["label"] = 1

# Combine datasets
data = pd.concat([fake_df, true_df])

# Create content column (same as training)
data["content"] = data["title"] + " " + data["text"]

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    # Get user input
    news = request.form.get("news", "").strip()

    # 🔥 If empty → fetch from dataset
    if news == "":
        random_row = data.sample(1)
        news = random_row["content"].values[0]

    # Transform input
    vector = vectorizer.transform([news])

    # Predict
    prediction = model.predict(vector)[0]

    # Output
    if prediction == 1:
        result = "REAL NEWS ✅"
    else:
        result = "FAKE NEWS ❌"

    return render_template("result.html", result=result)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)