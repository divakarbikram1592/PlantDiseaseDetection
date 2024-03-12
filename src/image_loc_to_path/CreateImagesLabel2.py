import os
import cv2

from src.FileUtils import FileUtils
from src.utils.MYCONSTANTS import MYCONSTANTS

"""
Modify code according to your folder and storage location
"""

proj_path = "/Users/apple/Documents/Projects-Python/PlantDisease/PlantVillageProject/"


################################################
#   Train data labels creation
################################################

path_1 = os.path.join(MYCONSTANTS.PROJ_PATH, "Potato___Early_blight")
path_2 = os.path.join(MYCONSTANTS.PROJ_PATH, "Potato___healthy")
path_3 = os.path.join(MYCONSTANTS.PROJ_PATH, "Potato___Late_blight")


fileutils = FileUtils()
train_data_list = fileutils.get_list(path=path_train)

train_file_path = os.path.join(MYCONSTANTS.PROJ_PATH, "train_labels.txt")
delimiter = ", "  # You can specify any delimiter you want
result_string = delimiter.join(train_data_list)
result = fileutils.write_file(result_string, train_file_path)
print(result)


################################################
#   Test data labels creation
################################################

path_test = os.path.join(MyConstants.PROJ_PATH, "test")
train_file_path = os.path.join(MyConstants.PROJ_PATH, "test_labels.txt")

fileutils = FileUtils()
train_data_list = fileutils.get_list(path=path_test)

delimiter = ", "  # You can specify any delimiter you want
result_string = delimiter.join(train_data_list)
result = fileutils.write_file(result_string, train_file_path)
print(result)
