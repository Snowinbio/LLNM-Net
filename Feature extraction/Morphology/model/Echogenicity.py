import os
import numpy as np
from skimage import io, transform, color, morphology as sm
from scipy.stats import norm
from scipy.optimize import minimize_scalar

class Echogenicity:
    def __init__(self, benign_img_dir, benign_edge_dir, malignant_img_dir, malignant_edge_dir, echo_train_bool):
        self.benign_img_dir = benign_img_dir
        self.benign_edge_dir = benign_edge_dir
        self.malignant_img_dir = malignant_img_dir
        self.malignant_edge_dir = malignant_edge_dir
        self.echo_train_bool = echo_train_bool
        self.img_name_list = []

    def get_nodule_echo(self, path, jpg_img, img_bound, dilation_value=3):
        img = io.imread(path)
        img = np.array(img, copy=False)
        img = transform.resize(image=img, output_shape=jpg_img.shape) * 255
        if img.ndim == 3:
            img = color.rgb2gray(img)
        img_dila = sm.dilation(img, sm.square(dilation_value))
        img_dila = (img_dila - np.min(img_dila)) / (np.max(img_dila) - np.min(img_dila)) * 255
        rows, cols, jpg_img_values, jpg_bound_values = [], [], [], []
        img_bound_values, back_bound_values = [], []

        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if img_dila[row, col] > 125 and img[row, col] < 125:
                    jpg_bound_values.append(jpg_img[row, col])
                    back_bound_values.append(img_bound[row, col])
                elif img[row, col] > 125:
                    rows.append(row)
                    cols.append(col)
                    jpg_img_values.append(jpg_img[row, col])
                    img_bound_values.append(img_bound[row, col])
                else:
                    back_bound_values.append(img_bound[row, col])

        if len(jpg_img_values) == 0:
            print(f"Warning: No nodule detected in {path}")
            return None

        nodule_echo = np.mean(jpg_img_values)
        bound_echo = np.mean(jpg_bound_values)
        echo_sub = bound_echo - nodule_echo
        return echo_sub

    def process_images(self, img_dir, edge_dir):
        echo_sub_list = []
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            edge_path = os.path.join(edge_dir, img_name)
            jpg_img = io.imread(img_path, as_gray=True)
            img_bound = io.imread(edge_path, as_gray=True)

            if jpg_img.shape != img_bound.shape:
                img_bound = transform.resize(img_bound, jpg_img.shape, mode='constant', preserve_range=True)
            echo_sub = self.get_nodule_echo(img_path, jpg_img, img_bound)
            if echo_sub is not None:
                echo_sub_list.append(echo_sub)
                self.img_name_list.append(img_name)

        return echo_sub_list

    def gaussian(self, x, mu, sigma):
        coefficient = 1 / (sigma * np.sqrt(2 * np.pi))  # 归一化系数
        exponent = - (x - mu) ** 2 / (2 * sigma ** 2)   # 指数部分
        return coefficient * np.exp(exponent)
    
    def diff_a_minus_b(self, x, mu_a, sigma_a, mu_b, sigma_b):
        a = self.gaussian(x, mu_a, sigma_a)
        b = self.gaussian(x, mu_b, sigma_b)
        return a - b
    
    def diff_b_minus_a(self, x, mu_a, sigma_a, mu_b, sigma_b):
        return -self.diff_a_minus_b(x, mu_a, sigma_a, mu_b, sigma_b)

    def find_max_diff(self, mu_a, sigma_a, mu_b, sigma_b, search_range):
        def neg_diff_func(x):
            return -self.diff_func(x, mu_a, sigma_a, mu_b, sigma_b)
        result = minimize_scalar(neg_diff_func, bounds=search_range, method='bounded')
        if result.success:
            max_x = result.x  # 最大值对应的x
            max_diff = self.diff_func(max_x, mu_a, sigma_a, mu_b, sigma_b)  # 最大差值
            return max_x, max_diff
        else:
            raise ValueError(f"Error in {result.message}")
        
    def calculate_range(self, mu_a, sigma_a, mu_b, sigma_b):
        search_left = min(mu_a, mu_b) - 3 * max(sigma_a, sigma_b)
        search_right = max(mu_a, mu_b) + 3 * max(sigma_a, sigma_b)
        search_range = (search_left, search_right)

        max_x_a_gt_b, max_diff_a_gt_b = self.find_max_diff(
            self.diff_a_minus_b, mu_a, sigma_a, mu_b, sigma_b, search_range
        )
        max_x_b_gt_a, max_diff_b_gt_a = self.find_max_diff(
            self.diff_b_minus_a, mu_a, sigma_a, mu_b, sigma_b, search_range
        )
        return max_diff_a_gt_b, max_diff_b_gt_a

    def calculate_statistics(self):
        benign_echo_sub_list = self.process_images(self.benign_img_dir, self.benign_edge_dir)
        malignant_echo_sub_list = self.process_images(self.malignant_img_dir, self.malignant_edge_dir)

        x_benign = np.array(benign_echo_sub_list)
        mu_benign = np.mean(x_benign)
        sigma_benign = np.std(x_benign)
        print("Benign mean echo_sub values:", mu_benign)
        print("Benign sigma echo_sub values:", sigma_benign)
        # if not self.echo_train_bool:

        x_malignant = np.array(malignant_echo_sub_list)
        mu_malignant = np.mean(x_malignant)
        sigma_malignant = np.std(x_malignant)
        print("Malignant mean echo_sub values:", mu_malignant)
        print("Malignant sigma echo_sub values:", sigma_malignant)

        if not self.echo_train_bool:
            mu_benign = -58.938271776546394
            sigma_benign = 18.219738470294324
            mu_malignant = -59.779479141140925
            sigma_malignant = 17.421576680834857
            self.benign_range = 0.00215678
            self.malignant_range = 0.00234567
        else:
            self.benign_range,self.malignant_range = self.calculate_range(mu_benign, sigma_benign, mu_malignant, sigma_malignant)

        return benign_echo_sub_list, malignant_echo_sub_list, mu_benign, sigma_benign, mu_malignant, sigma_malignant

    def calculate_p_norm(self):
        benign_echo_sub_list, malignant_echo_sub_list, mu_benign, sigma_benign, mu_malignant, sigma_malignant = self.calculate_statistics()

        echo_sub_list = benign_echo_sub_list + malignant_echo_sub_list
        p_norm_echo = []
        for i in echo_sub_list:
            p_malignant = norm.pdf(i, mu_malignant, sigma_malignant)
            p_benign = norm.pdf(i, mu_benign, sigma_benign)
            p_echo = p_malignant - p_benign
            if p_echo > 0:
                p_norm_echo.append(0.5+p_echo/(2*self.malignant_range))
            else:
                p_norm_echo.append(0.5+p_echo/(2*self.benign_range))

        p_norm_echo_dict = {}
        for i in range(len(self.img_name_list)):
            p_norm_echo_dict[self.img_name_list[i]] = p_norm_echo[i]

        return p_norm_echo_dict
