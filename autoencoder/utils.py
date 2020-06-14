

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from PIL import Image
from skimage.measure import compare_ssim as ssim
from termcolor import colored

from cifarTenDataset import CifarTenDataset

torch.manual_seed(0)


def create_dataloader(dataset_path: str="", val_size: float=0.05, test_size: float=0.05, batch_size: int=32):
    """ create three dataloader (train, validation, test)
    :param str dataset_path: path to dataset
    :param float val_size/test_size: validation-/test-percentage of dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, val- and test-dataloader
    """

    dataset = CifarTenDataset(root_dir=dataset_path, class_amount=3)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=1,
        shuffle=True
    )

    return dataloader


def calculate_ssim(y_true: torch.tensor, y_pred: torch.tensor) -> int:
    """ calculate the structural similarity of prediction and actual image
    
    :param torch.tensor y_true: targets
    :param torch.tensor y_pred: predictions
    :param float threshold: treshold for logit-rounding
    :return float: structural-similarity
    """

    total_ssim = 0
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]
        total_ssim += ssim(output.reshape(32, 32, 3), target.reshape(32, 32, 3), multichannel=True)
    
    return float(total_ssim / len(y_true))


def show_progress(epochs: int, epoch: int, train_loss: float, train_ssim: float, val_loss: float, val_ssim: float):
    """ print training stats
    
    :param int epochs: amount of total epochs
    :param int epoch: current epoch
    :param float train_loss/train_ssim: train-loss, train-ssim
    :param float val_loss/val_ssim: validation ssim/loss
    :return None
    """

    epochs = colored(epoch, "cyan", attrs=["bold"]) + colored("/", "cyan", attrs=["bold"]) + colored(epochs, "cyan", attrs=["bold"])
    train_ssim = colored(round(train_ssim, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
    train_loss = colored(round(train_loss, 6), "cyan", attrs=["bold"])
    val_ssim = colored(round(val_ssim, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
    val_loss = colored(round(val_loss, 6), "cyan", attrs=["bold"])
    
    print("epoch {} train_loss: {} - train_ssim: {} - val_loss: {} - val_ssim: {}".format(epochs, train_loss, train_ssim, val_loss, val_ssim), "\n")