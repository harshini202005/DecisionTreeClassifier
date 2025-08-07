from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['gpa']),
        float(request.form['extra']),
        float(request.form['hours']),
        int(request.form['exam']),
        int(request.form['internet'])
    ]
    final_features = np.array([features])
    prediction = model.predict(final_features)[0]
    return render_template('result.html', prediction_text=f'The predicted result is: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
