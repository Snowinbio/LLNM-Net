import os

from skimage import io, transform, color, morphology as sm
import numpy as np


class ShapeFeatureExtractor:
    def __init__(self, dilation_value=3):
        self.dilation_value = dilation_value

    def get_nodule_ratio(self, img):
        img_gray = np.array(img, copy=False)
        jpg_img = np.ones((256, 256))  
        img_resized = transform.resize(img_gray, output_shape=jpg_img.shape) * 255
        if img_resized.ndim == 3:
            img_resized = color.rgb2gray(img_resized)
        img_dila = sm.dilation(img_resized, sm.square(self.dilation_value))
        img_dila = (img_dila - np.min(img_dila)) / (np.max(img_dila) - np.min(img_dila)) * 255
        rows, cols = [], []
        for row in range(img_dila.shape[0]):
            for col in range(img_dila.shape[1]):
                if img_dila[row, col] > 125:
                    rows.append(row)
                    cols.append(col)
        if len(rows) == 0 or len(cols) == 0:
            return None
        row_max, row_min, col_max, col_min = max(rows), min(rows), max(cols), min(cols)
        aspect_ratio = (col_max - col_min + 1) / (row_max - row_min + 1)
        return aspect_ratio

    def process_directory(self, directory):
        ratios = []
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG"):
                img_path = os.path.join(directory, filename)
                img = io.imread(img_path)
                ratio = self.get_nodule_ratio(img)
                if ratio is not None:
                    ratios.append((filename, ratio))
                else:
                    print(f"No tumor detected in image: {filename}")
        return ratios

    def process_benign_and_malignant(self, benign_dir, malignant_dir):
        benign_ratios = self.process_directory(benign_dir)
        malignant_ratios = self.process_directory(malignant_dir)
        all_ratios = benign_ratios + malignant_ratios
        return all_ratios, len(benign_ratios), len(malignant_ratios)

    def normalize_ratios(self, ratios):
        ratios_values = np.array([ratio for _, ratio in ratios])
        min_ratio = np.min(ratios_values)
        max_ratio = np.max(ratios_values)
        if min_ratio == max_ratio:  
            return ratios

        normalized_ratios = [(filename, (ratio - min_ratio) / (max_ratio - min_ratio)) for filename, ratio in ratios]
        return normalized_ratios

    def print_ratios(self, all_ratios, num_benign, num_malignant):
        normalized_ratios = self.normalize_ratios(all_ratios)
        normalized_benign_ratios = normalized_ratios[:num_benign]
        normalized_malignant_ratios = normalized_ratios[num_benign:num_benign + num_malignant]

        aspect_ratios = [round(ratio, 2) for _, ratio in normalized_benign_ratios + normalized_malignant_ratios]

        return aspect_ratios




