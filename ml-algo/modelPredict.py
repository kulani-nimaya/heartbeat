import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Input data for prediction (raw values, no standardization)
input_data = np.array([[62, 92, 55, 1, 0]])  # HR, BT, Age, Smoke, FHCD

# Make prediction
outcome = model.predict(input_data)
print("Predicted Outcome:",outcome[0])