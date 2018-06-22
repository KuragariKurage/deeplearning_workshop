import cv2
import os
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

# x: data list
# y: label list
# image_dir: save images to this directory
# image_name_prefix: image names are "image_name_prefix1.tiff", "image_name_prefix2.tiff", ...
# dataset_file: save dataset file to this path
def generate(x, y, image_dir, image_name_prefix, dataset_file):

    f = open(dataset_file, "w")
    data_count = 0
    for data, label in zip(x, y):
        data_path = os.path.join(image_dir, image_name_prefix + str(data_count) + ".tiff")
        cv2.imwrite(data_path, data)
        f.write("{},{}".format(data_path, label))
        f.write("\n")
        data_count += 1

def main():
    image_dir = "data/image/"
    dataset_dir = "data/"
    val_rate = 0.1

    # check directory
    if os.path.isdir(image_dir) is False:
        os.makedirs(image_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # split train data to validation data
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=val_rate
    )

    # generate training data
    y_train = np.squeeze(y_train)
    train_dataset_path = os.path.join(dataset_dir, "train_dataset.txt")
    generate(x_train, y_train, image_dir, "train_", train_dataset_path)

    # generate validation data
    y_validation = np.squeeze(y_validation)
    validation_dataset_path = os.path.join(dataset_dir, "validation_dataset.txt")
    generate(x_validation, y_validation, image_dir, "validation_", validation_dataset_path)

    # generate test data
    y_test = np.squeeze(y_test)
    test_dataset_path = os.path.join(dataset_dir, "test_dataset.txt")
    generate(x_test, y_test, image_dir, "test_", test_dataset_path)


if __name__ == '__main__':
    main()
