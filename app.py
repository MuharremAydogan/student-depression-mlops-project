from flask import Flask, render_template, request
from src.pipeline.modelpredictPipeline import Predict
from main import main
app = Flask(__name__)

predictor = Predict()


@app.route("/")
def homepage():
    return render_template("index.html", prediction=None)

@app.route("/train")
def train():
    try:
        main()
        return "Training succesfuly"
    except Exception as e:
        raise e     

@app.route("/predict", methods=["POST"])
def predict():

    form = request.form

    data = [
        int(form["age"]),
        form["gender"],
        form["department"],
        float(form["cgpa"]),
        float(form["sleep"]),
        float(form["study"]),
        float(form["social"]),
        int(form["physical"]),
        int(form["stress"])
    ]

    y_pred = predictor.predict(data)
    prediction = int(y_pred[0])

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
