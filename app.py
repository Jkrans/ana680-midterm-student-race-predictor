from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    math_score = float(request.form["math_score"])
    reading_score = float(request.form["reading_score"])
    writing_score = float(request.form["writing_score"])

    X = np.array([[math_score, reading_score, writing_score]])
    pred = model.predict(X)[0]

    return render_template("index.html", result=pred)

if __name__ == "__main__":
    app.run(debug=True)
