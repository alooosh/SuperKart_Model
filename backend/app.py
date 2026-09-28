# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model and feature metadata
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
product_selling_predictor_api = Flask("Most Product Selling Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Load the expected feature names from training
model_features = joblib.load("model_features.joblib")

# Load the list of categorical columns used for one-hot encoding
categorical_cols = joblib.load("categorical_columns.joblib")

# Define a route for the home page (GET request)
@product_selling_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Most Product Selling Predictor"

# Define an endpoint for single product prediction (POST request)
@product_selling_predictor_api.post('/v1/predict')
def predict_product():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product details and returns
    the predicted product sales as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Convert the extracted data into a Pandas DataFrame
    input_df = pd.DataFrame([product_data])

    # Apply one-hot encoding to categorical features
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, dtype=int)

    # Reindex the DataFrame to match the model's training features
    # This adds any missing columns (with 0) and drops any extra columns
    input_df_aligned = input_df_encoded.reindex(columns=model_features, fill_value=0)

    # Make prediction
    predicted_sales = model.predict(input_df_aligned)[0]

    # Convert predicted_sales to Python float for jsonify
    predicted_sales = round(float(predicted_sales), 2)

    # Return the predicted sales
    return jsonify({'Predicted Product Sales (in dollars)': predicted_sales})

# Define an endpoint for batch prediction (POST request)
@product_selling_predictor_api.post('/v1/predictbatch')
def predict_rental_price_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product details for multiple products
    and returns the predicted product sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_df = pd.read_csv(file)

    # Apply one-hot encoding to categorical features
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, dtype=int)

    # Reindex the DataFrame to match the model's training features
    input_df_aligned = input_df_encoded.reindex(columns=model_features, fill_value=0)

    # Make predictions for all products in the DataFrame
    predicted_sales_batch = model.predict(input_df_aligned).tolist()

    # Convert predicted_sales_batch to Python floats
    predicted_sales_batch = [round(float(sales), 2) for sales in predicted_sales_batch]

    # Create a dictionary of predictions with row index as keys
    output_dict = dict(zip(input_df.index.tolist(), predicted_sales_batch))

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    # Note: In a production environment, use a production-ready WSGI server like Gunicorn
    product_selling_predictor_api.run(debug=True, host='0.0.0.0', port=7860)
