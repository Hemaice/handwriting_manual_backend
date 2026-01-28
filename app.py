from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from feature_extractor import ManualHandwritingFeatureExtractor

app = Flask(__name__)

MODEL_PATH = "handwriting_personality_model.pkl"
model = joblib.load(MODEL_PATH)

feature_extractor = ManualHandwritingFeatureExtractor()

@app.route('/')
def home():
    return jsonify({"message": "Handwriting Personality API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    os.makedirs("uploads", exist_ok=True)
    upload_path = os.path.join("uploads", file.filename)
    file.save(upload_path)

    try:
        features_dict = feature_extractor.extract_features(upload_path)
        if features_dict is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        feature_values = list(features_dict.values())
        feature_array = np.array(feature_values).reshape(1, -1)

        prediction = model.predict(feature_array)[0]

        return jsonify({
            "prediction": prediction,
            "features_used": list(features_dict.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
