import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class EncodeScale:
    def label_encode(param):
        # encode the target labels
        targetNames = np.unique(param)
        le = LabelEncoder()
        target = le.fit_transform(param)
        print("[STATUS] training labels encoded...")
        return target

    def minmax_scale(param):
        # normalize the feature vector in the range (0-1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_features = scaler.fit_transform(param)
        print("[STATUS] feature vector normalized...")
        return rescaled_features
