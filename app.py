from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np


# Load model
model = pickle.load(open('model2.pkl', 'rb'))
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('heartfailpredict.html')   

@app.route('/predict', methods=['POST'])
def predict_heart_failure():
    # Get data from form
    age = float(request.form.get('age'))
    anaemia = int(request.form.get('anaemia'))
    creatinine_phosphokinase = float(request.form.get('creatinine_phosphokinase'))
    diabetes = int(request.form.get('diabetes'))
    ejection_fraction = float(request.form.get('ejection_fraction'))
    high_blood_pressure = int(request.form.get('high_blood_pressure'))
    platelets = float(request.form.get('platelets'))
    serum_creatinine = float(request.form.get('serum_creatinine'))
    serum_sodium = float(request.form.get('serum_sodium'))
    sex = int(request.form.get('sex'))
    smoking = int(request.form.get('smoking'))
    time = float(request.form.get('time'))

    # Put into array for model
    input_features = np.array([[
        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time
    ]])

    # Predict
    prediction = model.predict(input_features)
    result = "Heart Failure Risk: YES (1)" if prediction[0] == 1 else "Heart Failure Risk: NO (0)"

    return render_template('heartfailpredict.html', result=result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
