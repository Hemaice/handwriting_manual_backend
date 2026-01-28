from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import tempfile
import os

# ========== LOAD MODELS ==========
feature_extractor = joblib.load("feature_extractor.pkl")
model = joblib.load("personality_model.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")
label_scaler = joblib.load("label_scaler.pkl")


# ========== FLASK APP ==========
app = Flask(__name__)
CORS(app)  # allow frontend (React) to call API


# ========== API ROUTE ==========
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check file exists
        if "file" not in request.files:
            return jsonify({"error": "No file found"}), 400

        file = request.files["file"]

        # Save temp file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Extract features
        features = feature_extractor.extract_features(file_path)
        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        # Convert to array (order must match training)
        feature_vector = np.array(list(features.values())).reshape(1, -1)

        # Scale features
        scaled_features = feature_scaler.transform(feature_vector)

        # Predict
        scaled_output = model.predict(scaled_features)

        # Convert back to original scale
        predictions = label_scaler.inverse_transform(scaled_output)[0]

        # Trait names (from training)
        trait_names = label_scaler.feature_names_in_.tolist()

        # Build traits dictionary
        trait_percentages = {
            trait_names[i]: float(predictions[i]) for i in range(len(predictions))
        }

        # Dominant trait
        dominant_trait = max(trait_percentages, key=trait_percentages.get)

        # Response to frontend
        return jsonify({
            "dominant_trait": dominant_trait,
            "traits": trait_percentages
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Handwriting Personality Prediction API is running!"


# ========== RUN SERVER ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
