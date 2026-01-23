# app.py - Complete Flask API for Render
import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from scipy import stats as sp_stats
from scipy.signal import find_peaks
from skimage.feature import graycomatrix, graycoprops
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS - Allow all origins for now, update with your React URL
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables for model
model_data = None

# Feature Extractor Class
class ManualHandwritingFeatureExtractor:
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, image_path):
        """Extract comprehensive features from a full-page handwriting scan"""
        try:
            # Read and preprocess image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            # Resize for consistency while maintaining aspect ratio
            height, width = img.shape
            new_height = 1000
            new_width = int(width * (new_height / height))
            img = cv2.resize(img, (new_width, new_height))
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(img, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            features = {}
            
            # 1. PAGE-LEVEL FEATURES
            features['page_width'] = new_width
            features['page_height'] = new_height
            features['aspect_ratio'] = new_width / new_height
            
            # 2. INK DENSITY FEATURES
            total_pixels = new_width * new_height
            ink_pixels = np.sum(binary > 0)
            
            features['global_ink_density'] = ink_pixels / total_pixels
            
            # Regional ink density (divide into 9 regions)
            h_parts = 3
            v_parts = 3
            region_densities = []
            for i in range(h_parts):
                for j in range(v_parts):
                    h_start = i * (new_height // h_parts)
                    h_end = (i + 1) * (new_height // h_parts)
                    w_start = j * (new_width // v_parts)
                    w_end = (j + 1) * (new_width // v_parts)
                    
                    region = binary[h_start:h_end, w_start:w_end]
                    region_density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
                    features[f'region_density_{i}_{j}'] = region_density
                    region_densities.append(region_density)
            
            features['density_variation'] = np.std(region_densities) if region_densities else 0
            
            if region_densities and len(region_densities) > 1:
                features['density_skew'] = sp_stats.skew(np.array(region_densities))
            else:
                features['density_skew'] = 0
            
            # 3. TEXT BLOCK FEATURES
            try:
                num_labels, labels, comp_stats, centroids = cv2.connectedComponentsWithStats(
                    binary, connectivity=8
                )
                
                if num_labels > 1:
                    component_areas = comp_stats[1:, cv2.CC_STAT_AREA]
                    large_components = component_areas[component_areas > 50]
                    
                    features['num_text_blocks'] = len(large_components)
                    features['avg_block_size'] = np.mean(large_components) if len(large_components) > 0 else 0
                    features['block_size_std'] = np.std(large_components) if len(large_components) > 0 else 0
                    features['block_size_ratio'] = np.sum(large_components) / total_pixels if len(large_components) > 0 else 0
                    
                    block_centroids = centroids[1:][component_areas > 50]
                    if len(block_centroids) > 0:
                        features['centroid_x_mean'] = np.mean(block_centroids[:, 0]) / new_width
                        features['centroid_y_mean'] = np.mean(block_centroids[:, 1]) / new_height
                        features['centroid_x_std'] = np.std(block_centroids[:, 0]) / new_width
                        features['centroid_y_std'] = np.std(block_centroids[:, 1]) / new_height
                    else:
                        features['centroid_x_mean'] = 0
                        features['centroid_y_mean'] = 0
                        features['centroid_x_std'] = 0
                        features['centroid_y_std'] = 0
                else:
                    features['num_text_blocks'] = 0
                    features['avg_block_size'] = 0
                    features['block_size_std'] = 0
                    features['block_size_ratio'] = 0
                    features['centroid_x_mean'] = 0
                    features['centroid_y_mean'] = 0
                    features['centroid_x_std'] = 0
                    features['centroid_y_std'] = 0
            except Exception as e:
                logger.error(f"Error in connected components: {e}")
                features['num_text_blocks'] = 0
                features['avg_block_size'] = 0
                features['block_size_std'] = 0
                features['block_size_ratio'] = 0
                features['centroid_x_mean'] = 0
                features['centroid_y_mean'] = 0
                features['centroid_x_std'] = 0
                features['centroid_y_std'] = 0
            
            # 4. LINE-LEVEL FEATURES
            horizontal_projection = np.sum(binary, axis=1)
            
            try:
                peaks, properties = find_peaks(horizontal_projection, 
                                             distance=30, 
                                             prominence=np.mean(horizontal_projection) * 0.3)
                
                if len(peaks) >= 2:
                    features['num_lines'] = len(peaks)
                    line_spacings = np.diff(peaks)
                    features['avg_line_spacing'] = np.mean(line_spacings)
                    features['line_spacing_std'] = np.std(line_spacings)
                    
                    line_straightness = []
                    for peak in peaks:
                        line_profile = binary[max(0, peak-10):min(new_height, peak+10), :]
                        if line_profile is not None and line_profile.size > 0:
                            col_sums = np.sum(line_profile, axis=0)
                            non_zero_cols = np.where(col_sums > 0)[0]
                            if len(non_zero_cols) > 1:
                                straightness = 1.0 / (1.0 + np.std(non_zero_cols))
                                line_straightness.append(straightness)
                    
                    features['avg_line_straightness'] = np.mean(line_straightness) if line_straightness else 0
                    features['line_straightness_std'] = np.std(line_straightness) if line_straightness else 0
                else:
                    features['num_lines'] = 0
                    features['avg_line_spacing'] = 0
                    features['line_spacing_std'] = 0
                    features['avg_line_straightness'] = 0
                    features['line_straightness_std'] = 0
            except Exception as e:
                logger.error(f"Error in line detection: {e}")
                features['num_lines'] = 0
                features['avg_line_spacing'] = 0
                features['line_spacing_std'] = 0
                features['avg_line_straightness'] = 0
                features['line_straightness_std'] = 0
            
            # 5. MARGIN FEATURES
            col_sums = np.sum(binary, axis=0)
            non_zero_cols = np.where(col_sums > 0)[0]
            
            if len(non_zero_cols) > 0:
                left_margin = non_zero_cols[0] / new_width
                right_margin = (new_width - non_zero_cols[-1]) / new_width
                text_width = (non_zero_cols[-1] - non_zero_cols[0]) / new_width
                
                features['left_margin'] = left_margin
                features['right_margin'] = right_margin
                features['text_width_ratio'] = text_width
                features['centeredness'] = abs(0.5 - (left_margin + text_width/2))
            else:
                features['left_margin'] = 0
                features['right_margin'] = 0
                features['text_width_ratio'] = 0
                features['centeredness'] = 0
            
            # 6. SLANT AND ANGLE FEATURES
            try:
                sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    angles = np.degrees(np.arctan2(sobel_y, sobel_x))
                    angles = angles[~np.isnan(angles)]
                    angles = angles[(angles > 10) & (angles < 170)]
                
                if len(angles) > 0:
                    features['slant_mean'] = np.mean(angles)
                    features['slant_std'] = np.std(angles)
                    features['slant_skew'] = sp_stats.skew(angles) if len(angles) > 1 else 0
                    
                    hist, bins = np.histogram(angles, bins=36, range=(0, 180))
                    dominant_slant = bins[np.argmax(hist)]
                    features['dominant_slant'] = dominant_slant
                    features['slant_uniformity'] = np.max(hist) / np.sum(hist) if np.sum(hist) > 0 else 0
                else:
                    features['slant_mean'] = 0
                    features['slant_std'] = 0
                    features['slant_skew'] = 0
                    features['dominant_slant'] = 0
                    features['slant_uniformity'] = 0
            except Exception as e:
                logger.error(f"Error in slant analysis: {e}")
                features['slant_mean'] = 0
                features['slant_std'] = 0
                features['slant_skew'] = 0
                features['dominant_slant'] = 0
                features['slant_uniformity'] = 0
            
            # 7. PRESSURE SIMULATION
            dark_pixels = img[img < 200]
            if len(dark_pixels) > 0:
                features['pressure_mean'] = np.mean(dark_pixels) / 255
                features['pressure_std'] = np.std(dark_pixels) / 255
                features['pressure_contrast'] = (np.max(dark_pixels) - np.min(dark_pixels)) / 255
            else:
                features['pressure_mean'] = 0
                features['pressure_std'] = 0
                features['pressure_contrast'] = 0
            
            # 8. TEXTURE FEATURES
            try:
                if img.shape[0] > 256 and img.shape[1] > 256:
                    texture_region = img[:256, :256]
                else:
                    texture_region = img
                
                glcm = graycomatrix(texture_region, distances=[1, 3], angles=[0, np.pi/4], 
                                   levels=256, symmetric=True, normed=True)
                
                for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                    try:
                        prop_value = graycoprops(glcm, prop)
                        features[f'texture_{prop}_mean'] = np.mean(prop_value)
                        features[f'texture_{prop}_std'] = np.std(prop_value)
                    except:
                        features[f'texture_{prop}_mean'] = 0
                        features[f'texture_{prop}_std'] = 0
            except Exception as e:
                logger.error(f"Error in texture analysis: {e}")
                for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                    features[f'texture_{prop}_mean'] = 0
                    features[f'texture_{prop}_std'] = 0
            
            # 9. EDGE AND CONTOUR FEATURES
            try:
                edges = cv2.Canny(img, 50, 150)
                features['edge_density'] = np.sum(edges > 0) / total_pixels
                
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour_areas = [cv2.contourArea(cnt) for cnt in contours]
                    contour_perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
                    
                    features['num_contours'] = len(contours)
                    features['contour_area_mean'] = np.mean(contour_areas)
                    features['contour_area_std'] = np.std(contour_areas)
                    
                    complexities = []
                    for p, a in zip(contour_perimeters, contour_areas):
                        if a > 0:
                            complexity = (p ** 2) / (4 * np.pi * a)
                            complexities.append(complexity)
                    
                    features['contour_complexity'] = np.mean(complexities) if complexities else 0
                else:
                    features['num_contours'] = 0
                    features['contour_area_mean'] = 0
                    features['contour_area_std'] = 0
                    features['contour_complexity'] = 0
            except Exception as e:
                logger.error(f"Error in edge/contour analysis: {e}")
                features['edge_density'] = 0
                features['num_contours'] = 0
                features['contour_area_mean'] = 0
                features['contour_area_std'] = 0
                features['contour_complexity'] = 0
            
            # 10. WHITE SPACE ANALYSIS
            try:
                row_white_space = []
                for row in range(0, new_height, 20):
                    row_slice = binary[row, :]
                    white_runs = []
                    in_white = False
                    run_length = 0
                    
                    for pixel in row_slice:
                        if pixel == 0:
                            if not in_white:
                                in_white = True
                                run_length = 1
                            else:
                                run_length += 1
                        else:
                            if in_white:
                                white_runs.append(run_length)
                                in_white = False
                    
                    if in_white:
                        white_runs.append(run_length)
                    
                    if white_runs:
                        row_white_space.append(np.mean(white_runs))
                
                if row_white_space:
                    features['avg_white_space'] = np.mean(row_white_space)
                    features['white_space_std'] = np.std(row_white_space)
                else:
                    features['avg_white_space'] = 0
                    features['white_space_std'] = 0
            except Exception as e:
                logger.error(f"Error in white space analysis: {e}")
                features['avg_white_space'] = 0
                features['white_space_std'] = 0
            
            # 11. ADDITIONAL FEATURES
            try:
                hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                features['entropy'] = entropy
            except:
                features['entropy'] = 0
            
            try:
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                features['gradient_mean'] = np.mean(gradient_magnitude)
                features['gradient_std'] = np.std(gradient_magnitude)
                features['gradient_skew'] = sp_stats.skew(gradient_magnitude.flatten()) if gradient_magnitude.size > 0 else 0
            except:
                features['gradient_mean'] = 0
                features['gradient_std'] = 0
                features['gradient_skew'] = 0
            
            vertical_projection = np.sum(binary, axis=0)
            features['vertical_projection_var'] = np.var(vertical_projection) / total_pixels if vertical_projection.size > 0 else 0
            
            # Ensure all values are finite
            for key in list(features.keys()):
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None

def load_model():
    """Load the trained model"""
    global model_data
    
    try:
        logger.info("Loading trained model...")
        
        # Try different paths for model file
        possible_paths = [
            'handwriting_personality_model.pkl',
            './handwriting_personality_model.pkl',
            os.path.join(os.path.dirname(__file__), 'handwriting_personality_model.pkl'),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            logger.error("Model file not found in any location")
            return False
        
        logger.info(f"Found model at: {model_path}")
        model_data = joblib.load(model_path)
        logger.info("✅ Model loaded successfully")
        
        # Verify model has required components
        required_keys = ['model', 'scaler', 'label_scaler', 'feature_columns', 'trait_names']
        for key in required_keys:
            if key not in model_data:
                logger.error(f"Missing key in model data: {key}")
                return False
        
        logger.info(f"Model type: {model_data.get('model_name', 'Random Forest')}")
        logger.info(f"Features: {len(model_data['feature_columns'])}")
        logger.info(f"Traits: {model_data['trait_names']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

# Load model on startup
load_model()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def categorize_trait(trait, score):
    """Categorize score into human-readable description"""
    if score >= 80:
        level = "Very High"
    elif score >= 60:
        level = "High"
    elif score >= 40:
        level = "Average"
    elif score >= 20:
        level = "Low"
    else:
        level = "Very Low"
    
    descriptions = {
        'openness': {
            'Very High': "Extremely creative, open-minded, curious",
            'High': "Creative, open to new experiences",
            'Average': "Moderately open, balanced",
            'Low': "Traditional, prefers routine",
            'Very Low': "Very traditional, resistant to change"
        },
        'conscientiousness': {
            'Very High': "Extremely organized, disciplined, reliable",
            'High': "Organized, responsible, diligent",
            'Average': "Moderately organized, balanced",
            'Low': "Spontaneous, flexible, casual",
            'Very Low': "Very spontaneous, disorganized"
        },
        'extraversion': {
            'Very High': "Extremely outgoing, sociable, energetic",
            'High': "Outgoing, enjoys social situations",
            'Average': "Balanced between social and solitary",
            'Low': "Reserved, prefers solitude",
            'Very Low': "Very introverted, avoids social situations"
        },
        'agreeableness': {
            'Very High': "Extremely compassionate, cooperative, trusting",
            'High': "Compassionate, cooperative, friendly",
            'Average': "Moderately agreeable, balanced",
            'Low': "Competitive, skeptical, direct",
            'Very Low': "Very challenging, critical, uncompromising"
        },
        'neuroticism': {
            'Very High': "Highly sensitive, anxious, emotionally reactive",
            'High': "Sensitive, occasionally anxious",
            'Average': "Moderately emotionally stable",
            'Low': "Calm, emotionally stable, resilient",
            'Very Low': "Extremely calm, emotionally stable, unflappable"
        }
    }
    
    return {
        'level': level,
        'description': descriptions[trait][level]
    }

# Routes
@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'Handwriting Personality Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Upload image for personality prediction',
            '/health': 'GET - Check API health',
            '/model-info': 'GET - Get model information'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'success',
        'message': 'API is running',
        'model_loaded': model_data is not None,
        'timestamp': np.datetime64('now').astype(str)
    })

@app.route('/model-info')
def model_info():
    if model_data is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    return jsonify({
        'status': 'success',
        'data': {
            'model_name': model_data.get('model_name', 'Random Forest'),
            'features_count': len(model_data['feature_columns']),
            'traits': model_data['trait_names'],
            'loaded': True,
            'sample_features': model_data['feature_columns'][:5] if len(model_data['feature_columns']) > 5 else model_data['feature_columns']
        }
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict personality from uploaded handwriting image"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 200
    
    if model_data is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please try again later.'
        }), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'status': 'error',
                'message': f'File too large. Maximum size is {MAX_FILE_SIZE/(1024*1024)}MB'
            }), 400
        
        logger.info(f"Processing file: {file.filename} ({file_size} bytes)")
        
        # Save file to temporary location
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        
        # Extract features
        extractor = ManualHandwritingFeatureExtractor()
        features = extractor.extract_features(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if features is None:
            return jsonify({
                'status': 'error',
                'message': 'Could not extract features from image. Please ensure it is a valid handwriting image.'
            }), 400
        
        # Prepare feature vector
        feature_vector = []
        missing_count = 0
        for col in model_data['feature_columns']:
            if col in features:
                feature_vector.append(features[col])
            else:
                feature_vector.append(0)
                missing_count += 1
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        features_scaled = model_data['scaler'].transform(feature_vector)
        
        # Predict
        predictions_scaled = model_data['model'].predict(features_scaled)
        
        # Inverse transform
        predictions = model_data['label_scaler'].inverse_transform(predictions_scaled)[0]
        
        # Clip to valid range
        predictions = np.clip(predictions, 0, 100)
        
        # Prepare results
        results = {}
        trait_names = model_data['trait_names']
        for i, trait in enumerate(trait_names):
            category_info = categorize_trait(trait, predictions[i])
            results[trait] = {
                'score': float(predictions[i]),
                'level': category_info['level'],
                'description': category_info['description'],
                'display_name': trait.capitalize()
            }
        
        # Find dominant and weakest traits
        scores = {trait: results[trait]['score'] for trait in results}
        dominant_trait = max(scores, key=scores.get)
        weakest_trait = min(scores, key=scores.get)
        
        # Calculate overall personality type
        avg_score = np.mean(list(scores.values()))
        if avg_score >= 70:
            overall_type = "Expressive & Dynamic"
        elif avg_score >= 55:
            overall_type = "Balanced & Adaptive"
        else:
            overall_type = "Reserved & Steady"
        
        logger.info(f"✅ Prediction successful for {file.filename}")
        logger.info(f"Dominant trait: {dominant_trait} ({scores[dominant_trait]:.1f})")
        logger.info(f"Weakest trait: {weakest_trait} ({scores[weakest_trait]:.1f})")
        
        return jsonify({
            'status': 'success',
            'message': 'Personality prediction successful',
            'data': {
                'traits': results,
                'summary': {
                    'dominant_trait': {
                        'name': dominant_trait,
                        'score': float(scores[dominant_trait]),
                        'display_name': dominant_trait.capitalize()
                    },
                    'weakest_trait': {
                        'name': weakest_trait,
                        'score': float(scores[weakest_trait]),
                        'display_name': weakest_trait.capitalize()
                    },
                    'overall_type': overall_type,
                    'average_score': float(avg_score)
                },
                'metadata': {
                    'model': model_data.get('model_name', 'Random Forest'),
                    'features_extracted': len(features),
                    'features_used': len(model_data['feature_columns']),
                    'missing_features': missing_count
                }
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'status': 'error',
        'message': f'File too large. Maximum size is {MAX_FILE_SIZE/(1024*1024)}MB'
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Handwriting Personality API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)