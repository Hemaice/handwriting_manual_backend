import cv2
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import find_peaks
from skimage.feature import graycomatrix, graycoprops
import logging

logger = logging.getLogger(__name__)

class ManualHandwritingFeatureExtractor:
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, image_path):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            height, width = img.shape
            new_height = 1000
            new_width = int(width * (new_height / height))
            img = cv2.resize(img, (new_width, new_height))
            
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            features = {}
            total_pixels = new_width * new_height

            features['page_width'] = new_width
            features['page_height'] = new_height
            features['aspect_ratio'] = new_width / new_height

            ink_pixels = np.sum(binary > 0)
            features['global_ink_density'] = ink_pixels / total_pixels
            
            region_densities = []
            for i in range(3):
                for j in range(3):
                    h_start = i * (new_height // 3)
                    h_end = (i + 1) * (new_height // 3)
                    w_start = j * (new_width // 3)
                    w_end = (j + 1) * (new_width // 3)
                    
                    region = binary[h_start:h_end, w_start:w_end]
                    density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
                    region_densities.append(density)
                    features[f'region_density_{i}_{j}'] = density
            
            features['density_variation'] = np.std(region_densities)
            features['density_skew'] = sp_stats.skew(region_densities)

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

            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
            angles = np.degrees(np.arctan2(sobel_y, sobel_x))
            angles = angles[(angles > 10) & (angles < 170)]

            if len(angles) > 0:
                features['slant_mean'] = np.mean(angles)
                features['slant_std'] = np.std(angles)
                features['slant_skew'] = sp_stats.skew(angles)
            else:
                features['slant_mean'] = 0
                features['slant_std'] = 0
                features['slant_skew'] = 0
            
            for k in list(features.keys()):
                if np.isnan(features[k]) or np.isinf(features[k]):
                    features[k] = 0
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
