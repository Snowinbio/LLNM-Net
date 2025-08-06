import cv2 as cv
import numpy as np
import random
import pickle
import os
import csv
from PIL import Image


class BscanObj:
    def __init__(self, label, img, seg):
        self.label = label
        self.img = img
        self.seg = seg

    def extract_edge_points(self):
        edgePot = []
        for i in range(1, self.seg.shape[0] - 1):
            for j in range(1, self.seg.shape[1] - 1):
                if self.seg[i, j] == 255 and ((int(self.seg[i, j - 1]) * int(self.seg[i, j + 1]) * int(self.seg[i - 1, j]) * int(self.seg[i + 1, j])) == 0):
                    edgePot.append((i, j))
        return list(set(edgePot))

    def sample_edge_points(self, edgePot, num_samples):
        if len(edgePot) < num_samples:
            randPot = [random.choice(edgePot) for _ in range(num_samples)]
        else:
            randPot = random.sample(edgePot, num_samples)
        return randPot

    def extract_features(self, randPot, feature_size=16):
        randFeat = []
        for k in randPot:
            minFeatPot = (k[0] - feature_size // 2, k[1] - feature_size // 2)
            maxFeatPot = (k[0] + feature_size // 2, k[1] + feature_size // 2)
            minFeatPot = (max(0, minFeatPot[0]), max(0, minFeatPot[1]))
            maxFeatPot = (min(self.seg.shape[0], maxFeatPot[0]), min(self.seg.shape[1], maxFeatPot[1]))

            featMap = self.img[minFeatPot[0]:maxFeatPot[0], minFeatPot[1]:maxFeatPot[1]]

            if featMap.size == 0:
                continue
            if featMap.shape[0] != feature_size or featMap.shape[1] != feature_size:
                featMap = cv.resize(featMap, (feature_size, feature_size))

            randFeat.append(featMap)
        return randFeat

    def combine_features(self, randFeat, num_rows, num_cols, feature_size=16):
        allFeat = np.zeros((num_rows * feature_size, num_cols * feature_size), dtype=randFeat[0].dtype)
        for idx, feat in enumerate(randFeat):
            row_idx = idx // num_cols
            col_idx = idx % num_cols
            allFeat[row_idx * feature_size:(row_idx + 1) * feature_size,
            col_idx * feature_size:(col_idx + 1) * feature_size] = feat
        return allFeat

    def edgeFeat_224x224(self):
        edgePot = self.extract_edge_points()
        randPot = self.sample_edge_points(edgePot, 196)
        randFeat = self.extract_features(randPot)
        return self.combine_features(randFeat, 14, 14)


class EdgeFeatureExtractor:
    def __init__(self, img_dir, seg_dir, save_dir):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.save_dir = save_dir

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def process_image(self, img_path, seg_path, save_path):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        seg = cv.imread(seg_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Failed to read image from {img_path}")
            return
        if seg is None:
            print(f"Error: Failed to read segmentation from {seg_path}")
            return
        Bscan = BscanObj(1, img, seg)
        edgeFeat = Bscan.edgeFeat_224x224()
        cv.imwrite(save_path, edgeFeat)

    def extract_and_save(self, datasets, categories):
        for dataset in datasets:
            for category in categories:
                print(f'Processing {dataset} set, category: {category}')
                category_img_dir = os.path.join(self.img_dir, dataset, category)
                category_seg_dir = os.path.join(self.seg_dir, dataset, category)
                category_save_dir = os.path.join(self.save_dir, dataset, category)

                self.ensure_dir(category_save_dir)
                for filename in os.listdir(category_img_dir):
                    # print(f'Processing file: {filename}')
                    img_path = os.path.join(category_img_dir, filename)
                    seg_path = os.path.join(category_seg_dir, filename)
                    save_path = os.path.join(category_save_dir, filename)

                    self.process_image(img_path, seg_path, save_path)

    @staticmethod
    def save_to_pickle(data, save_path):
        with open(save_path, 'wb') as write_file:
            pickle.dump(data, write_file)


class Detection:
    def __init__(self, img_dir, det_dir, dst_dir):
        self.img_dir = img_dir
        self.det_dir = det_dir
        self.dst_dir = dst_dir

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def detect_and_crop(self):
        img_list = os.listdir(self.img_dir)
        box_list = []
        for img_name in img_list:
            im = Image.open(os.path.join(self.img_dir, img_name))
            det_img_name = img_name
            de = Image.open(os.path.join(self.det_dir, det_img_name))
            det = np.array(de)

            nz = np.nonzero(det)
            row_min = int(np.min(nz[0]))
            row_max = int(np.max(nz[0]))
            col_min = int(np.min(nz[1]))
            col_max = int(np.max(nz[1]))

            box = (col_min, row_min, col_max, row_max)
            im_c = im.crop(box)
            im_c.save(os.path.join(self.dst_dir, det_img_name))

            im_box = (img_name, col_min, row_min, col_max, row_max)
            box_list.append(im_box)

        with open(os.path.join(self.dst_dir, "box.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            for box_info in box_list:
                writer.writerow(box_info)

        return 0