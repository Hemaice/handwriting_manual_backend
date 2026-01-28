import os
import joblib
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from feature_extractor import ManualHandwritingFeatureExtractor

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
extractor = ManualHandwritingFeatureExtractor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

logger.info("Loading model...")
model = joblib.load("handwriting_personality_model.pkl")
logger.info("Model loaded successfully.")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("/tmp", filename)
    file.save(filepath)

    features = extractor.extract_features(filepath)
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    input_vector = [features[k] for k in sorted(features.keys())]

    pred = model.predict([input_vector])[0]
    return jsonify({"prediction": pred}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
