

import torchvision
import torch
import numpy as np


class CifarTenDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str="", class_amount: int=10):
        """ constructor
        :param str root_dir: dataset path
        :param int class_amount: amount of classes in the dataset
        """

        self.root_dir = root_dir
        self.class_amount = class_amount
        self.dataset = np.load(self.root_dir, allow_pickle=True)
        
    def _create_one_hot(self, int_representation: int) -> list:
        """ create one-hot encoding of the target
        :param int int_representation: class of sample
        :return list: ie. int_representation = 2 -> [0, 0, 1, ..., 0]
        """

        one_hot_target = np.zeros((self.class_amount))
        one_hot_target[int_representation] = 1
        return one_hot_target

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ get sample (batch) from dataset
        :param int idx: index of dataset (iterator of training-loop)
        :return tensor: preprocessed sample and target
        """

        # not using the target
        sample, _ = self.dataset[idx][0], self.dataset[idx][1]

        sample = sample.reshape(3, 32, 32)
        sample = sample / 255
        
        return torch.Tensor(sample)

    def __len__(self):
        """ returns length of dataset """
        
        return len(self.dataset)