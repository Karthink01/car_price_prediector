from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Load the model
with open(r'C:\Users\Admin\Downloads\001\001\LinearRegressionModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the car data
car = pd.read_csv(r'C:\Users\Admin\Downloads\001\001\Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        # Log the received input
        app.logger.info(f"Received input - Company: {company}, Car Model: {car_model}, Year: {year}, Fuel Type: {fuel_type}, Kms Driven: {driven}")

        input_data = pd.DataFrame({
            'name': [car_model],
            'company': [company],
            'year': [year],
            'kms_driven': [driven],
            'fuel_type': [fuel_type]
        })

        # Log the input data for prediction
        app.logger.info(f"Input data for prediction: {input_data}")

        prediction = model.predict(input_data)
        app.logger.info(f"Prediction result: {prediction}")

        return jsonify({'prediction': np.round(prediction[0], 2)})
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
