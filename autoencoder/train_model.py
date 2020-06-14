
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import torch
import torch.utils.data
import torch.nn as nn
import cv2

from model import Model
from utils import calculate_ssim, show_progress, create_dataloader

torch.manual_seed(0)


class Run:
    def __init__(self, dataset_paths: list=[], model_path: str="", epochs: int=10, lr: float=0.001, batch_size: int=16):
        self.dataset_paths = dataset_paths
        self.model_path = model_path

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.train_set = create_dataloader(dataset_path=self.dataset_paths[0], batch_size=self.batch_size)
        self.validation_set = create_dataloader(dataset_path=self.dataset_paths[1], batch_size=self.batch_size)
        self.test_set = create_dataloader(dataset_path=self.dataset_paths[2], batch_size=self.batch_size)

    def _validate(self, model, dataset):
        validation_dataset = dataset

        criterion = nn.MSELoss()

        losses = []
        total_targets, total_predictions = [], []
        for images in tqdm(validation_dataset, desc="validating", ncols=150):
            images = images.cuda()
            predictions = model.eval()(images, train_=False)

            loss = criterion(images, predictions)
            losses.append(loss.item())

            for i in range(predictions.size()[0]):
                total_targets.append(images[i].cpu().detach().numpy().reshape(32, 32, 3))
                total_predictions.append(predictions[i].cpu().detach().numpy().reshape(32, 32, 3))

        # calculate ssim
        ssim = calculate_ssim(total_targets, total_predictions)

        # calculate loss
        loss = np.mean(losses)
        return loss, ssim

    def train(self, continue_: bool=False):
        model = Model().cuda()
        if continue_:
            model.load_state_dict(torch.load(self.model_path))
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        train_loss, train_ssim, val_loss, val_ssim = [], [], [], []
        for epoch in range(1, (self.epochs + 1)):
            
            epoch_loss = []
            epoch_train_targets, epoch_train_predictions = [], []
            for images in tqdm(self.train_set, desc="epoch", ncols=150):
                optimizer.zero_grad()

                images = images.float().cuda()
                predictions = model.train()(images, train_=True)

                loss = criterion(predictions, images)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

                for i in range(predictions.size()[0]):
                    epoch_train_targets.append(images[i].cpu().detach().numpy()); \
                    epoch_train_predictions.append(predictions[i].cpu().detach().numpy())

            current_val_loss, current_val_ssim = self._validate(model, self.validation_set)
            current_train_loss = np.mean(epoch_loss)
            current_train_ssim = calculate_ssim(epoch_train_targets, epoch_train_predictions)

            show_progress(self.epochs, epoch, current_train_loss, current_train_ssim, current_val_loss, current_val_ssim)

            torch.save(model.state_dict(), self.model_path)

    def test(self, show: bool=True):
        model = Model().cuda()
        model.load_state_dict(torch.load(self.model_path))

        _, ssim = self._validate(model, self.test_set)
        print("average train-ssim:", round(ssim, 5))

        if show:
            for images in self.test_set:
                images = images.float().cuda()
                reconstructed_images = model.eval()(images, train_=False)

                for idx in range(images.size(0)):
                    image = images.cpu().detach().numpy()[idx].reshape(32, 32, 3)
                    reconstructed = reconstructed_images.cpu().detach().numpy()[idx].reshape(32, 32, 3)

                    fig, axs = plt.subplots(2)
                    axs[0].imshow(image)
                    axs[1].imshow(reconstructed)
                    plt.show()
    
    def create_bottleneck_dataset(self, save_file: str=""):
        """ saves all bottlenecks from the train-set images to k-means-clustering/dataset/

        :param str save_file: path to the dataset-folder in the k-means-clustering/
        """

        model = Model().cuda()
        model.load_state_dict(torch.load(self.model_path))

        train_bottlenecks = []
        """for images in tqdm(self.train_set, desc="epoch", ncols=150):
            images = images.float().cuda()

            bottlenecks = model.eval()(images, train_=False, return_bottlenecks=True)

            for idx in range(len(bottlenecks)):
                bottleneck = bottlenecks[idx].cpu().detach().numpy()
                train_bottlenecks.append(bottleneck)"""
        for sample in tqdm(np.load(self.dataset_paths[0], allow_pickle=True)):
            image = sample[0] / 255
            image = torch.Tensor(image).reshape(1, 3, 32, 32).cuda()

            bottleneck = model.eval()(image, train_=False, return_bottlenecks=True)[0]
            bottleneck = bottleneck.cpu().detach().numpy()

            train_bottlenecks.append(bottleneck)

        np.save(save_file, np.array(train_bottlenecks))

run = Run(dataset_paths=["datasets/train_dataset.npy", "datasets/val_dataset.npy", "datasets/test_dataset.npy"],
            model_path="models/model2.pt",
            epochs=100,
            lr=0.00075,
            batch_size=32)


#run.train(continue_=True)
#run.test(show=False)
run.create_bottleneck_dataset(save_file="../k-means-classification/dataset/train_bottlenecks.npy")


