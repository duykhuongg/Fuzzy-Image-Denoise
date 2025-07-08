import numpy as np
import cv2
from fuzzy import Fuzzy

class MedianFuzzyFilter:
    def __init__(self):
        self.fuzzy = Fuzzy()
        
        self.DL_params = {'c': 0, 'sigma': 30}
        self.DH_params = {'c': 255, 'sigma': 30}
        
        self.VL_params = {'a': 0, 'b': 0, 'c': 0.2, 'd': 0.4}
        self.VH_params = {'a': 0.6, 'b': 0.8, 'c': 1, 'd': 1}
        
        self.noise_threshold = 80
        self.edge_threshold = 30

    def get_sub_windows_indices(self):
        indices = [
            ([0, 1, 2, 5, 6, 7, 10, 11, 12], 1.5),    # Cửa sổ trên trái
            ([1, 2, 3, 6, 7, 8, 11, 12, 13], 1.3),    # Cửa sổ trên giữa
            ([2, 3, 4, 7, 8, 9, 12, 13, 14], 1.0),    # Cửa sổ trên phải
            ([5, 6, 7, 10, 11, 12, 15, 16, 17], 1.3), # Cửa sổ giữa trái
            ([7, 8, 9, 12, 13, 14, 17, 18, 19], 1.0), # Cửa sổ giữa phải
            ([10, 11, 12, 15, 16, 17, 20, 21, 22], 1.0), # Cửa sổ dưới trái
            ([11, 12, 13, 16, 17, 18, 21, 22, 23], 1.3), # Cửa sổ dưới giữa
            ([12, 13, 14, 17, 18, 19, 22, 23, 24], 1.0)  # Cửa sổ dưới phải
        ]
        return indices

    def preprocess_window(self, window):
        flat = window.flatten()
        q1, q3 = np.percentile(flat, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        mask = (flat < lower) | (flat > upper)
        if np.any(mask):
            median = np.median(flat[~mask])
            flat[mask] = median
        
        return flat.reshape(window.shape)

    def calculate_medians(self, window):
        flat_window = window.flatten()
        center_value = flat_window[12]
        
        medians = []
        weights = []
        indices_list = self.get_sub_windows_indices()
        
        for sub_indices, weight in indices_list:
            sub_values = [flat_window[i] for i in sub_indices]
            medians.append(np.median(sub_values))
            weights.append(weight)
        
        return medians, center_value, weights

    def fuzzy_inference(self, differences, weights):

        memberships_low = []
        memberships_high = []
        
        for d, w in zip(differences, weights):
            ml = self.fuzzy.gaussmf(d, self.DL_params['c'], 
                                   self.DL_params['sigma']) * w
            mh = self.fuzzy.gaussmf(d, self.DH_params['c'], 
                                   self.DH_params['sigma']) * w
            memberships_low.append(ml)
            memberships_high.append(mh)
        
        firing_VL = min(memberships_low)
        
        firing_VH = max(memberships_high)
        
        x_output = np.linspace(0, 1, 1000)
        
        vl = self.fuzzy.trapmf(x_output, **self.VL_params) 
        vh = self.fuzzy.trapmf(x_output, **self.VH_params)
        
        vl_activated = np.minimum(vl, firing_VL)
        vh_activated = np.minimum(vh, firing_VH)
        
        combined_mu = np.maximum(vl_activated, vh_activated)
        
        noisiness = self.fuzzy.defuzz_centroid(x_output, combined_mu)
        
        return noisiness

    def process_pixel(self, window):
        medians, center_value, weights = self.calculate_medians(window)
        differences = [abs(m - center_value) for m in medians]
        
        noisiness = self.fuzzy_inference(differences, weights)
        
        if noisiness > 0.8:
            center_dist = [abs(i-12) for i in range(25)]
            weights = [1/(d+1) for d in center_dist]
            valid_medians = []
            valid_weights = []
            
            for m, d, w in zip(medians, differences, weights):
                if d < self.noise_threshold:
                    valid_medians.append(m)
                    valid_weights.append(w)
                    
            if valid_medians:
                reconstructed_value = np.average(valid_medians, weights=valid_weights)
            else:
                reconstructed_value = np.median(medians)
        else:
            weights = [1/(d + 1e-6) * w for d, w in zip(differences, weights)]
            reconstructed_value = np.average(medians, weights=weights)
        
        new_value = noisiness * reconstructed_value + (1 - noisiness) * center_value
        return int(round(new_value))

    def non_max_suppression(self, mag, angle):
        height, width = mag.shape
        result = np.zeros_like(mag)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                angle_i = angle[i,j]
                if angle_i < 0:
                    angle_i += 180
                    
                if (0 <= angle_i < 22.5) or (157.5 <= angle_i <= 180):
                    prev = mag[i, j-1]
                    next = mag[i, j+1]
                elif 22.5 <= angle_i < 67.5:
                    prev = mag[i+1, j-1]
                    next = mag[i-1, j+1]
                elif 67.5 <= angle_i < 112.5:
                    prev = mag[i+1, j]
                    next = mag[i-1, j]
                else:
                    prev = mag[i-1, j-1]
                    next = mag[i+1, j+1]
                    
                if mag[i,j] >= prev and mag[i,j] >= next:
                    result[i,j] = mag[i,j]
                    
        return result

    def detect_edges(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi
        suppressed = self.non_max_suppression(magnitude, angle)
        
        return suppressed > self.edge_threshold

    def filter_image(self, image):
        padded = np.pad(image, ((2, 2), (2, 2)), mode='reflect')
        result = np.zeros_like(image)
        
        edges = self.detect_edges(image)
        
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                if not edges[i, j]:
                    window = padded[i:i+5, j:j+5]
                    result[i, j] = self.process_pixel(window)
                else:
                    result[i, j] = image[i, j]
        
        padded_second = np.pad(result, ((2, 2), (2, 2)), mode='reflect')
        final_result = np.zeros_like(image)
        
        edges_second = self.detect_edges(result)
        self.noise_threshold *= 0.7
        
        for i in range(rows):
            for j in range(cols):
                if not edges_second[i, j]:
                    window = padded_second[i:i+5, j:j+5]
                    final_result[i, j] = self.process_pixel(window)
                else:
                    final_result[i, j] = result[i, j]
                
        self.noise_threshold /= 0.7
        
        return final_result 