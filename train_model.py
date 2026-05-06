import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("🚀 Training started...")

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Assign labels
fake['label'] = 0   # FAKE
true['label'] = 1   # REAL

# Combine
data = pd.concat([fake, true])

# Shuffle dataset (VERY IMPORTANT)
data = data.sample(frac=1, random_state=42)

# Combine text
data['content'] = data['title'] + " " + data['text']

# Features & target
X = data['content']
y = data['label']

# Remove nulls
data = data.dropna()

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vector = vectorizer.fit_transform(X)

# Split (STRATIFIED - VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42, stratify=y
)

# Model (BALANCED)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Debug predictions
print("Sample predictions:", model.predict(X_test[:10]))
print("Actual:", y_test[:10].values)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model saved successfully!")