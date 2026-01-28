# feature_extractor.py
import cv2
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import find_peaks
from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')

class ManualHandwritingFeatureExtractor:
    def __init__(self):
        self.feature_names = []

    def extract_features(self, image_path):
        """Extract comprehensive features from a full-page handwriting scan"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # Resize
            height, width = img.shape
            new_height = 1000
            new_width = int(width * (new_height / height))
            img = cv2.resize(img, (new_width, new_height))

            # Threshold
            binary = cv2.adaptiveThreshold(img, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)

            features = {}

            # Page-level
            features['page_width'] = new_width
            features['page_height'] = new_height
            features['aspect_ratio'] = new_width / new_height

            # Ink density
            total_pixels = new_width * new_height
            ink_pixels = np.sum(binary > 0)
            features['global_ink_density'] = ink_pixels / total_pixels

            # Regional ink density (3x3)
            region_densities = []
            for i in range(3):
                for j in range(3):
                    h_start = i * (new_height // 3)
                    h_end = (i + 1) * (new_height // 3)
                    w_start = j * (new_width // 3)
                    w_end = (j + 1) * (new_width // 3)
                    region = binary[h_start:h_end, w_start:w_end]
                    region_density = np.sum(region > 0) / (region.shape[0]*region.shape[1])
                    features[f'region_density_{i}_{j}'] = region_density
                    region_densities.append(region_density)

            features['density_variation'] = np.std(region_densities) if region_densities else 0
            features['density_skew'] = sp_stats.skew(np.array(region_densities)) if len(region_densities) > 1 else 0

            # Connected components for text blocks
            try:
                num_labels, labels, comp_stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                if num_labels > 1:
                    component_areas = comp_stats[1:, cv2.CC_STAT_AREA]
                    large_components = component_areas[component_areas > 50]
                    features['num_text_blocks'] = len(large_components)
                    features['avg_block_size'] = np.mean(large_components) if len(large_components) > 0 else 0
                    features['block_size_std'] = np.std(large_components) if len(large_components) > 0 else 0
                    features['block_size_ratio'] = np.sum(large_components)/total_pixels if len(large_components)>0 else 0

                    block_centroids = centroids[1:][component_areas > 50]
                    if len(block_centroids) > 0:
                        features['centroid_x_mean'] = np.mean(block_centroids[:, 0])/new_width
                        features['centroid_y_mean'] = np.mean(block_centroids[:, 1])/new_height
                        features['centroid_x_std'] = np.std(block_centroids[:,0])/new_width
                        features['centroid_y_std'] = np.std(block_centroids[:,1])/new_height
                    else:
                        features['centroid_x_mean'] = 0
                        features['centroid_y_mean'] = 0
                        features['centroid_x_std'] = 0
                        features['centroid_y_std'] = 0
                else:
                    features.update({k:0 for k in ['num_text_blocks','avg_block_size','block_size_std','block_size_ratio','centroid_x_mean','centroid_y_mean','centroid_x_std','centroid_y_std']})
            except:
                features.update({k:0 for k in ['num_text_blocks','avg_block_size','block_size_std','block_size_ratio','centroid_x_mean','centroid_y_mean','centroid_x_std','centroid_y_std']})

            # Line-level, margin, slant, pressure, texture, edge, white space, entropy etc.
            # For brevity, copy your full implementation here as is from Colab.
            # Ensure all keys exist and NaNs are replaced with 0

            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return None
