import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Product Sales Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product features
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66, step=0.01)
Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ['Low Sugar', 'Regular', 'No Sugar', 'reg'],
    index=0 # Default to 'Low Sugar'
)
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027, step=0.001)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08, step=0.01)
Store_Establishment_Year = st.number_input("Store Establishment Year", min_value=1900, value=2009, step=1)
Store_Size = st.selectbox(
    "Store Size",
    ['Medium', 'High', 'Small'],
    index=0 # Default to 'Medium'
)
Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ['Tier 2', 'Tier 1', 'Tier 3'],
    index=0 # Default to 'Tier 2'
)
Store_Type = st.selectbox(
    "Store Type",
    ['Supermarket Type2', 'Supermarket Type1', 'Departmental Store', 'Food Mart'],
    index=0 # Default to 'Supermarket Type2'
)
Product_Type = st.selectbox(
    "Product Type",
    ['Fruits and Vegetables', 'Snack Foods', 'Frozen Foods', 'Dairy', 'Household', 'Baking Goods', 'Canned', 'Health and Hygiene', 'Meat', 'Soft Drinks', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood'],
    index=0 # Default to 'Fruits and Vegetables'
)

# Create a dictionary payload for the API request
input_payload = {
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': int(Store_Establishment_Year),
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type,
    'Product_Type': Product_Type
}

# Make prediction when the "Predict" button is clicked
if st.button("Predict Sales", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_payload)
    if response.status_code == 200:
        prediction = response.json()['Predicted Product Sales (in dollars)']
        st.success(f"Predicted Product Sales (in dollars): {prediction}")
    else:
        st.error(f"Error connecting to the prediction API: {response.status_code} - {response.text}")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch Sales", type="primary"):
        # Ensure the file is reset to the beginning if already read
        uploaded_file.seek(0)
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            # Display predictions in a more readable format, e.g., a DataFrame
            st.dataframe(pd.DataFrame(predictions.items(), columns=['Original Index', 'Predicted Sales (in dollars)']))
        else:
            st.error(f"Error connecting to the batch prediction API: {response.status_code} - {response.text}")
