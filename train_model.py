# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/StudentsPerformance.csv")

X = df[["math score", "reading score", "writing score"]].copy()
y = df["race/ethnicity"]

# Numeric safety
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())
X = X.clip(0, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=10000,
        solver="saga",
        C=0.5
    ))
])

pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", round(acc, 4))

with open("model.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Saved model.pkl")
