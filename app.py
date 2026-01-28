from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from feature_extractor import ManualHandwritingFeatureExtractor

# Load pickles
model = joblib.load("personality_model.pkl")
scaler = joblib.load("feature_scaler.pkl")
label_scaler = joblib.load("label_scaler.pkl")
extractor = joblib.load("feature_extractor.pkl")  # works because class is imported

# Feature and trait names (must match training)
feature_columns = [...]  # list of all features used in training
trait_columns = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_path = os.path.join("/tmp", file.filename)
    file.save(image_path)

    # Extract features
    features = extractor.extract_features(image_path)
    if features is None:
        return jsonify({"error": "Could not extract features"}), 500

    # Convert to DataFrame
    df = pd.DataFrame([features])
    df = df[feature_columns]  # ensure correct order

    # Scale features
    X_scaled = scaler.transform(df)

    # Predict
    y_pred_scaled = model.predict(X_scaled)

    # Reverse label scaling
    y_pred = label_scaler.inverse_transform(y_pred_scaled)

    # Prepare response
    result = {trait: float(y_pred[0, i]) for i, trait in enumerate(trait_columns)}
    return jsonify(result)

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
