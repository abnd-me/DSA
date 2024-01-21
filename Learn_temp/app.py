from flask import Flask, render_template,request
import pickle


app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/prediction', methods = ['GET' , 'POST'])
def predict():
    if request.method == 'POST':
        height = request.form['height']
        print('height')
        model = pickle.load(open('model.pkl', 'rb'))
        weight = model.predict([[float(height)]])
        print(weight)
    return render_template('prediction.html', weight=weight)


if __name__ == '__main__':
    app.run()