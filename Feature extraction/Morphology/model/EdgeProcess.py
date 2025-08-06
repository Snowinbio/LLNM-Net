import cv2 as cv
import numpy as np
import random
import pickle
import os


class BscanObj():
    def __init__(self, label, img, seg):
        self.label = label
        self.img = img
        self.seg = seg

    def edgeFeat_64x64(self):
        edgePot = []
        for i in range(1,self.seg.shape[0]-1):
            for j in range(1, self.seg.shape[1]-1):
                if self.seg[i,j] == 255 and (self.seg[i,j-1] * self.seg[i,j+1] * self.seg[i-1,j] * self.seg[i+1,j] == 0):
                    edgePot.append((i,j))

        edgePot = list(set(edgePot))
        if len(edgePot) < 16:
            randPot = [random.choice(edgePot) for _ in range(16)]
        else:
            randPot = random.sample(edgePot, 16)

        randFeat = []
        for k in randPot:
            minFeatPot = (k[0]-8, k[1]-8)
            maxFeatPot = (k[0]+8, k[1]+8)
            if minFeatPot[0] < 0:
                minFeatPot = (0, minFeatPot[1])
                maxFeatPot = (16, maxFeatPot[1])
            if minFeatPot[1] < 0:
                minFeatPot = (minFeatPot[0], 0)
                maxFeatPot = (maxFeatPot[0], 16)
            if maxFeatPot[0] > self.seg.shape[0]:
                minFeatPot = (self.seg.shape[0]-16, minFeatPot[1])
                maxFeatPot = (self.seg.shape[0], maxFeatPot[1])
            if maxFeatPot[1] > self.seg.shape[1]:
                minFeatPot = (minFeatPot[0], self.seg.shape[1]-16)
                maxFeatPot = (maxFeatPot[0], self.seg.shape[1])
            featMap = self.img[minFeatPot[0]:maxFeatPot[0], minFeatPot[1]:maxFeatPot[1]]
            randFeat.append(featMap)

        l1 = np.concatenate((randFeat[0], randFeat[1], randFeat[2], randFeat[3]), axis=1)
        l2 = np.concatenate((randFeat[4], randFeat[5], randFeat[6], randFeat[7]), axis=1)
        l3 = np.concatenate((randFeat[8], randFeat[9], randFeat[10], randFeat[11]), axis=1)
        l4 = np.concatenate((randFeat[12], randFeat[13], randFeat[14], randFeat[15]), axis=1)
        allFeat = np.concatenate((l1,l2,l3,l4), axis=0)
        return allFeat

    def edgeFeat_128x128(self):
        edgePot = []
        for i in range(1, self.seg.shape[0] - 1):
            for j in range(1, self.seg.shape[1] - 1):
                if self.seg[i, j] == 255 and ((int(self.seg[i, j - 1]) * int(self.seg[i, j + 1]) * int(self.seg[i - 1, j]) * int(self.seg[i + 1, j])) == 0):
                    edgePot.append((i, j))

        edgePot = list(set(edgePot))
        if len(edgePot) < 64:
            randPot = [random.choice(edgePot) for _ in range(64)]
        else:
            randPot = random.sample(edgePot, 64)
        randFeat = []

        for k in randPot:
            minFeatPot = (k[0] - 8, k[1] - 8)
            maxFeatPot = (k[0] + 8, k[1] + 8)
            if minFeatPot[0] < 0:
                minFeatPot = (0, minFeatPot[1])
                maxFeatPot = (16, maxFeatPot[1])
            if minFeatPot[1] < 0:
                minFeatPot = (minFeatPot[0], 0)
                maxFeatPot = (maxFeatPot[0], 16)
            if maxFeatPot[0] > self.seg.shape[0]:
                minFeatPot = (self.seg.shape[0] - 16, minFeatPot[1])
                maxFeatPot = (self.seg.shape[0], maxFeatPot[1])
            if maxFeatPot[1] > self.seg.shape[1]:
                minFeatPot = (minFeatPot[0], self.seg.shape[1] - 16)
                maxFeatPot = (maxFeatPot[0], self.seg.shape[1])
            featMap = self.img[minFeatPot[0]:maxFeatPot[0], minFeatPot[1]:maxFeatPot[1]]
            randFeat.append(featMap)

        for a in range(8):
            row = randFeat[a*8]
            for b in range(1,8):
                row = np.concatenate((row, randFeat[a*8+b]), axis=1)
            if a == 0:
                allFeat = row
            else:
                allFeat = np.concatenate((allFeat, row), axis=0)

        return allFeat

    def edgeFeat_224x224(self):
        edgePot = []
        for i in range(1, self.seg.shape[0] - 1):
            for j in range(1, self.seg.shape[1] - 1):
                if self.seg[i, j] == 255 and ((int(self.seg[i, j - 1]) * int(self.seg[i, j + 1]) * int(self.seg[i - 1, j]) * int(self.seg[i + 1, j])) == 0):
                    edgePot.append((i, j))

        edgePot = list(set(edgePot))
        if len(edgePot) < 196:
            randPot = [random.choice(edgePot) for _ in range(196)]
        else:
            randPot = random.sample(edgePot, 196)

        randFeat = []
        for k in randPot:
            minFeatPot = (k[0] - 8, k[1] - 8)
            maxFeatPot = (k[0] + 8, k[1] + 8)
            if minFeatPot[0] < 0:
                minFeatPot = (0, minFeatPot[1])
                maxFeatPot = (16, maxFeatPot[1])
            if minFeatPot[1] < 0:
                minFeatPot = (minFeatPot[0], 0)
                maxFeatPot = (maxFeatPot[0], 16)
            if maxFeatPot[0] > self.seg.shape[0]:
                minFeatPot = (self.seg.shape[0] - 16, minFeatPot[1])
                maxFeatPot = (self.seg.shape[0], maxFeatPot[1])
            if maxFeatPot[1] > self.seg.shape[1]:
                minFeatPot = (minFeatPot[0], self.seg.shape[1] - 16)
                maxFeatPot = (maxFeatPot[0], self.seg.shape[1])
            featMap = self.img[minFeatPot[0]:maxFeatPot[0], minFeatPot[1]:maxFeatPot[1]]

            if featMap.size == 0:
                continue
            feature_size = 16
            if featMap.shape[0] != feature_size or featMap.shape[1] != feature_size:
                featMap = cv.resize(featMap, (feature_size, feature_size))

            randFeat.append(featMap)

        # for a in range(14):
        #     row = randFeat[a * 14]
        #     for b in range(1, 14):
        #         row = np.concatenate((row, randFeat[a * 14 + b]), axis=1)
        #     if a == 0:
        #         allFeat = row
        #     else:
        #         allFeat = np.concatenate((allFeat, row), axis=0)

        num_rows = 14
        num_cols = 14
        feature_size = 16
        allFeat = np.zeros((num_rows * feature_size, num_cols * feature_size), dtype=randFeat[0].dtype)
        for idx, feat in enumerate(randFeat):
            row_idx = idx // num_cols
            col_idx = idx % num_cols
            allFeat[row_idx * feature_size:(row_idx + 1) * feature_size,
            col_idx * feature_size:(col_idx + 1) * feature_size] = feat

        return allFeat

def label_ex(labelDir0, labelDir1):
    labelD = {}
    for i in os.listdir(labelDir0):
        labelD[i] = 0
    for i in os.listdir(labelDir1):
        labelD[i] = 1
    return labelD

def edgeFeat_ex(imgDir, segDir, savePath):
    selected_mali_folders = ['benign', 'malignant']
    for mali in selected_mali_folders:
        if not os.path.exists(os.path.join(savePath, mali)):
            os.makedirs(os.path.join(savePath, mali))
        for i in os.listdir(os.path.join(imgDir, mali)):
            # print('fileName:', i)
            img = cv.imread(os.path.join(imgDir, mali, i).encode('gbk').decode('gbk'), cv.IMREAD_GRAYSCALE)
            seg = cv.imread(os.path.join(segDir, mali, i).encode('gbk').decode('gbk'), cv.IMREAD_GRAYSCALE)
            Bscan = BscanObj(1, img, seg)
            edgeFeat = Bscan.edgeFeat_224x224()
            cv.imwrite(os.path.join(savePath, mali, i), edgeFeat)
    return 0

def pkl_save(trainData, savePath):
    with open(savePath, 'wb') as writeFile:
        pickle.dump(trainData, writeFile)

def dec_process(img_dir, label_dir, edge_dir):
    # imgDir = r"./data/image/train"
    # segDir = r"./data/label/train"
    # savePath = r"./data/edge_images/train"

    edgeFeat_ex(img_dir, label_dir, edge_dir)
