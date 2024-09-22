from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('purchase_prediction_model.h5')

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json()

        # Parse features from the input data
        age = data['age']
        gender = data['gender']
        time_on_site = data['time_on_site']
        past_purchases = data['past_purchases']
        cart_items = data['cart_items']

        # Combine the data into a numpy array for prediction
        features = np.array([[age, gender, time_on_site, past_purchases, cart_items]])

        # Normalize the features using the same scaler used in training
        features = scaler.transform(features)

        # Make the prediction
        purchase_prob = model.predict(features)
        purchase_decision = (purchase_prob > 0.5).astype(int)

        # Prepare the response
        response = {
            'purchase_probability': float(purchase_prob[0][0]),
            'purchase_decision': int(purchase_decision[0][0])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app locally
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Change host/port if needed
