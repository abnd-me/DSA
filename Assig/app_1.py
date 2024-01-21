from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle


app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
#model = joblib.load('d_model.joblib')

# model = joblib.load('d_model.joblib')



# Render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input from the form
            sl = float(request.form['sl'])
            sw = float(request.form['sw'])
            pl = float(request.form['pl'])
            pw = float(request.form['pw'])

            # Make a prediction using the loaded model
            prediction = model.predict([[sl, sw, pl, pw]])

            # Return the predicted species
            result = {'prediction': prediction[0]}
            return render_template('result.html', result=result)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
