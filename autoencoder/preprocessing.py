
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



data_batches = ["datasets/cifar-10/data_batch_" + str(i) for i in range(1, 6)]  # + [("datasets/cifar-10/" + "test_batch")]

classes = {"airplane": 0, "dog": 5, "truck": 9}
targets = dict(zip(list(classes.values()), list(range(len(classes)))))

def unpickle(file: str=""):
    """ read pickle file

    :param str file: path to pickle-file
    :return dict: pickle-file content
    """

    with open(file, "rb") as fo:
        dict_ = pickle.load(fo, encoding="bytes")
    return dict_

def preprocess_data(batch: list, labels: list):
    """ read elements from dataset and convert to image format, save with labels
    
    :param list batch: sample-batch
    :param list labels: label-batch
    :return list: preprocessed dataset-batch
    """
    
    dataset = []
    for i in tqdm(range(len(batch))):
        if labels[i] in list(classes.values()):
            test_sample = data[i]

            r, b, g = test_sample[:1024], test_sample[1024:2048], test_sample[2048:]
            img = list(zip(r, b, g))
            img = np.array(img).reshape(32, 32, 3)

            dataset.append(np.array([img, targets[labels[i]]]))
    
    return dataset

def split(dataset: list, testing_size: float=0.1, validation_size: float=0.1):
    test_size = int(np.round(len(dataset)*testing_size))
    val_size = int(np.round(len(dataset)*validation_size))
    train_set, validation_set, test_set = dataset[(test_size+val_size):], dataset[:test_size], dataset[test_size:(test_size+val_size)]

    return train_set, validation_set, test_set

if __name__ == "__main__":
    final_dataset = []
    for batch in data_batches:
        dataset = unpickle(batch)
        data = dataset[b"data"]
        labels = dataset[b"labels"]

        dataset = preprocess_data(data, labels)
        final_dataset += dataset

    train_set, val_set, test_set = split(final_dataset, testing_size=0.075, validation_size=0.075)

    np.save("datasets/train_dataset.npy", np.array(train_set))
    np.save("datasets/val_dataset.npy", np.array(val_set))
    np.save("datasets/test_dataset.npy", np.array(test_set))
