from flask import Flask, render_template, request
import pickle

# Create Flask app
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    # Read values from the form
    life = float(request.form["life"])
    school = float(request.form["school"])
    gni = float(request.form["gni"])

    # Predict HDI
    result = model.predict([[life, school, gni]])

    # Show result
    return render_template(
        "result.html",
        prediction=round(result[0], 3)
    )

# Run the website
if __name__ == "__main__":
    app.run(debug=True)