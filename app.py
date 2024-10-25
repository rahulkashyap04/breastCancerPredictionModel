from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# loading model
model = pickle.load(open('model.pkl', 'rb'))

# flask app
app = Flask(__name__)


@app.route('/')

def home():
    return render_template('cancerModel.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.form['feature']
        features_list = features.split(',')
        np_feature = np.asarray(features_list, dtype=np.float32)

        # Correct the variable name here
        pred = model.predict(np_feature.reshape(1, -1))  # Change np_features to np_feature
        output = ['Cancers' if pred[0] == 1 else 'Not cancers']
        return render_template('cancerModel.html', message=output)
    except Exception as e:
        return render_template('cancerModel.html', message=[f"Error: {str(e)}"])


if __name__ == '__main__':
    app.run(debug=True)
