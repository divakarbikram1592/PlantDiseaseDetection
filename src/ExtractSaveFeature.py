import numpy as np
import h5py
import os
# from MYCONSTANTS import MYCONSTANTS
from ImageFeatureExtraction import ImageFeatureExtraction
from EncodeScale import EncodeScale
from FileUtils import FileUtils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class ExtractSaveFeature:

    def get_train(path):
        h5_data_path = "resource/output/data.h5"
        h5_label_path = "resource/output/labels.h5"

        if os.path.exists(h5_data_path) and os.path.exists(h5_label_path):
            # import the feature vector and trained labels
            h5f_data = h5py.File(h5_data_path, 'r')
            h5f_label = h5py.File(h5_label_path, 'r')

            global_features_string = h5f_data['dataset_1']
            global_labels_string = h5f_label['dataset_1']

            rescaled_features = np.array(global_features_string)
            target = np.array(global_labels_string)

            h5f_data.close()
            h5f_label.close()

            return rescaled_features, target
        else:
            train_global_feature, train_labels = ImageFeatureExtraction().get_feature(path)

            rescaled_features = EncodeScale.minmax_scale(train_global_feature)
            target = EncodeScale.label_encode(train_labels)
            print("===target===>", target)
            # save target and feature in h5
            FileUtils().save_in_h5(rescaled_features, target)

            return rescaled_features, target

    def get_test(path):
        test_global_feature, test_labels = ImageFeatureExtraction().get_feature(path)

        rescaled_features = EncodeScale.minmax_scale(test_global_feature)
        target = EncodeScale.label_encode(test_labels)

        testDataGlobal = np.array(rescaled_features)
        testLabelsGlobal = np.array(target)
        return testDataGlobal, testLabelsGlobal




