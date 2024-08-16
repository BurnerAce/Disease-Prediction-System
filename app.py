from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('best_model.pkl') 
scaler = joblib.load('scaler.pkl') 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        chest_pain_type = float(request.form['chest_pain_type'])
        blood_pressure = float(request.form['blood_pressure'])
        cholesterol = float(request.form['cholesterol'])
        fasting_blood_sugar = float(request.form['fasting_blood_sugar'])
        resting_ecg = float(request.form['resting_ecg'])
        max_heart_rate = float(request.form['max_heart_rate'])
        exercise_angina = float(request.form['exercise_angina'])
        oldpeak = float(request.form['oldpeak'])
        st_slope = float(request.form['st_slope'])
        input_data = np.array([[age, sex, chest_pain_type, blood_pressure, cholesterol, fasting_blood_sugar, 
                                resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        if prediction == 1:
            message = "The prediction indicates a high likelihood of disease."
        else:
            message = "The prediction indicates a low likelihood of disease."

        return render_template('result.html', message=message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
