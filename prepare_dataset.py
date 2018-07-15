import os
from collections import defaultdict
from sklearn.datasets import load_files
from keras.utils import np_utils
from PIL import Image
import numpy as np

data_dir = "flowers"
min_image_size = 128

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
        print(ret[-1].shape)
    ret = np.asarray(ret, dtype=np.float32)
    return ret

if __name__ == '__main__':

    flower_files, flower_targets = load_dataset(data_dir)
    input_array = files_to_array(flower_files)
    print(flower_targets)
