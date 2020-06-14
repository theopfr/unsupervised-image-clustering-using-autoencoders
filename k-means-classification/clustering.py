

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import entropy


class KMeans:
    def __init__(self, dataset_path: str="", train_images_path: str="", classes: int=3):
        self.dataset_path = dataset_path
        self.dataset = np.load(self.dataset_path, allow_pickle=True)

        self.images = np.load(train_images_path, allow_pickle=True)

        self.classes = classes
        self.dims = self.dataset[0].shape

        # normalize between [0, 1]
        self.dataset = self.dataset / np.max(self.dataset)

    def _get_distance(self, a: list, b: list) -> float:
        """ euclidean distance """

        between = np.subtract(b, a)
        distance = np.sqrt(np.sum(pow(between, 2)))
        return distance

    def _get_nearest_ks(self, Ks: list, points: list):
        """ creates list of length |points| which contains for every index of each point the nearest K-point """

        k_per_point = []
        for i in range(len(points)):

            nearest_k = 0
            nearest_distance = self._get_distance(Ks[0], points[i])
            for j in range(len(Ks)):
                current_distance = self._get_distance(Ks[j], points[i])
                
                if current_distance < nearest_distance:
                    nearest_distance = current_distance
                    nearest_k = j
            
            k_per_point.append(nearest_k)

        return k_per_point

    def _update_means(self, Ks: list, points: list, k_per_point: list) -> list:
        """ update k-coordinates """

        new_Ks = []
        for i in range(len(Ks)):
            new_K = np.zeros(self.dims)
            total_points_of_k = 0

            for j in range(len(points)):
                if k_per_point[j] == i:
                    new_K += points[j]
                    total_points_of_k += 1
                        
            if total_points_of_k == 0:
                new_Ks.append(Ks[i])
            else:
                new_K = new_K / total_points_of_k
                new_Ks.append(new_K)
        
        return new_Ks

    def cluster(self, steps: int=10):
        """ create k-means cluster """

        rand_low = np.mean(self.dataset) - self.dataset.std()
        rand_high = np.mean(self.dataset) + self.dataset.std()

        Ks = [np.random.uniform(rand_low, rand_high, self.dims) for _ in range(self.classes)]

        # clustering
        points = self.dataset
        k_per_point = self._get_nearest_ks(Ks, points)
        for step in tqdm(range(steps)):
            Ks = self._update_means(Ks, points, k_per_point)
            k_per_point = self._get_nearest_ks(Ks, points)
    
        clusters = [[] for _ in range(self.classes)]
        for k in range(len(Ks)):
            for i in range(len(points)):
                if k_per_point[i] == k:
                    clusters[k].append(i)

        return clusters
        
    def supervise_cluster(self, clusters: list=[], c: int=0):
        """ plots image cluster, looks at the labels for the first time, calculates entropy of labels in a cluster c """

        cluster = clusters[c]

        class_distributions = [0 for _ in range(self.classes)]
        for i in range(len(cluster)):
            class_ = self.images[cluster[i]][1]
            
            for j in range(len(class_distributions)):
                if j == class_:
                    class_distributions[j] += 1
                    break

        print("class-distributions:", class_distributions)
        print("entropy:", entropy(class_distributions))

        cluster = cluster[:144]
        fig, axs = plt.subplots(12, 12)
        counter = 0
        for i in range(12):
            for j in range(12):
                axs[i][j].imshow(self.images[cluster[counter]][0])
                counter += 1

        plt.show()

kMeans = KMeans(dataset_path="dataset/train_bottlenecks.npy", train_images_path="../autoencoder/datasets/train_dataset.npy", classes=3)
clusters = kMeans.cluster(steps=10)

kMeans.supervise_cluster(clusters=clusters, c=2)