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

# Expected raw feature columns for incoming CSV/JSON payloads
required_input_columns = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Type'
]

ignored_input_columns = ['Store_Establishment_Year', 'Product_Type']

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
    try:
     # Get the JSON data from the request body
     product_data = request.get_json()

     if not isinstance(product_data, dict):
         return jsonify({'Error': 'Request body must be a JSON object.'}), 400

     missing_cols = [col for col in required_input_columns if col not in product_data]
     if missing_cols:
         return jsonify({'Error': f'Missing required input fields: {missing_cols}'}), 400

     # Convert the extracted data into a Pandas DataFrame
     input_df = pd.DataFrame([product_data])

     # Ignore columns that are not part of the active feature set for prediction
     input_df = input_df.drop(columns=ignored_input_columns, errors='ignore')

     # Apply one-hot encoding only to categorical columns that still exist in the input
     categorical_columns_present = [col for col in categorical_cols if col in input_df.columns]
     input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns_present, dtype=int)

     # Reindex the DataFrame to match the model's training features
     # This adds any missing columns (with 0) and drops any extra columns
     input_df_aligned = input_df_encoded.reindex(columns=model_features, fill_value=0)

     # Make prediction
     predicted_sales = model.predict(input_df_aligned)[0]

     # Convert predicted_sales to Python float for jsonify
     predicted_sales = round(float(predicted_sales), 2)

     # Return the predicted sales
     return jsonify({'Predicted Product Sales (in dollars)': predicted_sales})
    except Exception as e:
     return jsonify({'Error': str(e)}), 500

# Define an endpoint for batch prediction (POST request)
@product_selling_predictor_api.post('/v1/predictbatch')
def predict_rental_price_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product details for multiple products
    and returns the predicted product sales as a dictionary in the JSON response.
    """
    try:
        # Get the uploaded CSV file from the request
        file = request.files.get('file')

        if file is None:
            return jsonify({'Error': 'No CSV file uploaded. Please upload a file under the "file" field.'}), 400

        # Read the CSV file into a Pandas DataFrame
        input_df = pd.read_csv(file)

        if input_df.empty:
            return jsonify({'Error': 'Uploaded CSV is empty.'}), 400


        # Ignore fields that should not affect prediction in this version of the model
        input_df = input_df.drop(columns=ignored_input_columns, errors='ignore')

        # Reorder only the actual model columns used for inference
        input_df = input_df[[col for col in required_input_columns if col not in ignored_input_columns]]

        # Apply one-hot encoding only to categorical columns that still exist in the input
        categorical_columns_present = [col for col in categorical_cols if col in input_df.columns]
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns_present, dtype=int)

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
    except Exception as e:
        return jsonify({'Error': str(e)}), 500

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    # Note: In a production environment, use a production-ready WSGI server like Gunicorn
    product_selling_predictor_api.run(debug=True, host='0.0.0.0', port=7860)
