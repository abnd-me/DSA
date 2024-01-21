from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
#with open('species_prediction_model.pkl', 'rb') as model_file:
 #   model = pickle.load(model_file)

model = pickle.load(open('species_prediction_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
        
        sl_value = float(request.form['sepal_length'])

        prediction = model.predict([[sl_value]])[0]

        return render_template('prediction.html', sepal_length=sl_value, prediction=prediction)
    

if __name__ == '__main__':
    app.run(debug=True)
