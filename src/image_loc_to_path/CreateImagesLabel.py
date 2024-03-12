import os
import cv2
import numpy as np

from src.FileUtils import FileUtils
from src.utils.MYCONSTANTS import MYCONSTANTS

"""
Modify code according to your folder and storage location
"""

proj_path = "/Users/apple/Documents/Projects-Python/PlantDisease/PlantVillageProject/"

folders = ["Potato___Early_blight", "Potato___healthy", "Potato___Late_blight"]
################################################
#   Train data labels creation
################################################
train_list = []
test_list = []
test_size = .33

def create_file(path, data):
    fileutils = FileUtils()
    result = fileutils.write_file(data, path)
    return result


def create_label(name):
    print("=========>", name)
    full_path = proj_path+name
    file_list = os.listdir(full_path)
    file_list = [os.path.join(full_path+"/", filename) for filename in file_list]
    # print(file_list)

    max_len = len(file_list)
    train_len = round(max_len * (1-test_size))

    # add images path in train(67% images) & test(33% images) list,
    train_list.append(file_list[0:train_len])
    test_list.append(file_list[train_len+1:max_len])



for name in folders:
    create_label(name)



# train
# 1
arr = np.array(train_list[0], dtype=str)
arr = arr.reshape(1, arr.shape[0])
# 2
arr1 = np.array(train_list[1], dtype=str)
arr1 = arr1.reshape(1, arr1.shape[0])
# 3
arr2 = np.array(train_list[2], dtype=str)
arr2 = arr2.reshape(1, arr2.shape[0])

#  3 list(arr, arr1, arr2) for 3 classes(Potato___Early_blight, Potato___healthy, Potato___Late_blight)
print(arr[0][0])
print(arr1[0][0])
print(arr2[0][0])


# Concatenate the arrays
train_result_array = np.concatenate((arr, arr1, arr2), axis=1)
# Set print options to display the full array
np.set_printoptions(threshold=np.inf)
# print(np.array(train_result_array))
array_string1 = np.array2string(train_result_array, separator=", ")
array_string_without_brackets2 = array_string1[2:-2]
# print(array_string_without_brackets2)
create_file("../resource/train_labels.txt", array_string_without_brackets2)

# test
# 1
tarr = np.array(test_list[0], dtype=str)
tarr = tarr.reshape(1, tarr.shape[0])
# 2
tarr1 = np.array(test_list[1], dtype=str)
tarr1 = tarr1.reshape(1, tarr1.shape[0])
# 3
tarr2 = np.array(test_list[2], dtype=str)
tarr2 = tarr2.reshape(1, tarr2.shape[0])

# Concatenate the arrays
test_result_array = np.concatenate((tarr, tarr1, tarr2), axis=1)
# print(test_result_array.shape)
array_string = np.array2string(test_result_array, separator=", ")
array_string_without_brackets = array_string[2:-2]
# print(array_string_without_brackets)
create_file("../resource/test_labels.txt", array_string_without_brackets)