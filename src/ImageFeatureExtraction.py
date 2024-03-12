import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import mahotas
import os
# from MYCONSTANTS import MYCONSTANTS
from FileUtils import FileUtils
import pandas as pd

# bins for histogram
bins = 8

class ImageFeatureExtraction:

    def __init__(self):
        self = self
    def calculatehistogram(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, 16, 2, method="uniform")
        (histogram, _) = np.histogram(lbp.ravel(),
                                      bins=np.arange(0, 16 + 3),
                                      range=(0, 16 + 2))
        # now we need to normalise the histogram so that the total sum=1
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + eps)
        return histogram

    # feature-descriptor-1: Hu Moments
    def fd_hu_moments(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        # print(len(feature))
        return feature

    # feature-descriptor-2: Haralick Texture
    def fd_haralick(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        # print(len(haralick))
        return haralick

    # feature-descriptor-3: Color Histogram
    def fd_histogram(self, image, mask=None):
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist = cv2.calcHist([image], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histogram
        # print(len(hist.flatten()))
        return hist.flatten()

    def fd_LBP(self, image):
        # print(">>>>>>>>>>>>>>>>>>>>", image.shape)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = self.calculatehistogram(gray)
        # kp = np.array(kp)
        # print(len(pts.flatten()))
        # print(len(hist.flatten()))
        return hist.flatten()

    def get_feature(self, path):
        train_labels = FileUtils().read_file(path).split(",")
        train_labels = [x.replace("\n", "").replace("'", "") for x in train_labels]
        # print(train_labels)
        # print("\n\n**train_labels", train_labels)
        global_feature = []
        labels = []
        total = len(train_labels)
        print("total", total)
        k = 0
        for label in train_labels:
            if len(label) < 20:
                return
            k = k + 1
            print("No. of processed", k,"/",total)
            # print(str(label))
            image = cv2.imread(label.strip())

            if image.any():
                hu_moments = self.fd_hu_moments(image)
                haralick = self.fd_haralick(image)
                histogram = self.fd_histogram(image)
                lbp = self.fd_LBP(image)

                # global_feature = np.stack([hu_moments, haralick, histogram, lbp], axis=2)
                _feature = np.hstack([hu_moments, haralick, histogram, lbp])
                global_feature.append(_feature)
                labels.append(label.split("/")[7])

        return global_feature, labels

