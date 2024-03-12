import os
import numpy as np
import h5py
class FileUtils:

    def __init__(self):
        print("")

    def get_list(self, path):

        file_list = []
        path1 = os.listdir(path)
        path1.sort()
        if path1[0] == ".DS_Store":
            path1.remove(".DS_Store")
        for i in range(len(path1)):
            path2 = os.path.join(path, path1[i])
            path3 = os.listdir(path2)
            for j in range(len(path3)):
                path4 = os.path.join(path2, path3[j])
                file_list.append(path4)

        return file_list

    def write_file(self, data, file_name):
        # Open the file in write mode
        with open(file_name, "w") as file:
            # Write some content to the file
            file.write(data)

        # The file is automatically closed when the "with" block exits
        return f"{file_name} has been created and written to."

    def read_file(self, file_path):
        # Open the file in write mode
        with open(file_path, "r") as file:
            data = file.read()
        return data


    def save_in_h5(self, rescaled_features, target):
        # h5_path = MYCONSTANTS.PROJ_PATH
        # save the feature vector using HDF5
        h5f_data = h5py.File('resource/output/data.h5', 'w')
        h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

        h5f_label = h5py.File('resource/output/labels.h5', 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(target))

        print("Data saved in h5")
        h5f_data.close()
        h5f_label.close()





