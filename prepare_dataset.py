import os
from collections import defaultdict
from sklearn.datasets import load_files
from keras.utils import np_utils
from PIL import Image
import numpy as np
import random

data_dir = "flowers"
min_image_size = 224

def get_train_test_sets(input_array, flower_targets):
    data_size = input_array.shape[0]
    sample_size = data_size // 10
    test_indexes = random.sample(list(range(data_size)), sample_size)
    training_indexes = [i for i in list(range(data_size)) if i not in test_indexes]
    training_data = input_array[training_indexes]
    training_output = flower_targets[training_indexes]
    testing_data = input_array[test_indexes]
    testing_output = flower_targets[test_indexes]
    return training_data, training_output, testing_data, testing_output

def load_dataset(path):
    data = load_files(path)
    flower_files = np.array(data["filenames"])
    flower_targets = np_utils.to_categorical(np.array(data["target"]), 5)
    return flower_files, flower_targets

def files_to_array(flower_files):
    ret = []
    for image in flower_files:
        im = Image.open(image)
        im = im.resize((min_image_size, min_image_size))
        ret.append(np.array(im, dtype=np.float32))
    ret = np.asarray(ret, dtype=np.float32)
    return ret

def fetch_training_data():
    flower_files, flower_targets = load_dataset(data_dir)
    input_array = files_to_array(flower_files)
    training_data, training_output, testing_data, testing_output = \
            get_train_test_sets(input_array, flower_targets)
    return training_data, training_output, testing_data, testing_output

if __name__ == '__main__':
    flower_files, flower_targets = load_dataset(data_dir)
    input_array = files_to_array(flower_files)
    training_data, training_output, testing_data, testing_output = \
            get_train_test_sets(input_array, flower_targets)
    print(training_data.shape, testing_data.shape)
