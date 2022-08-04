import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from train import compute_hog
from utils import load_checkpoint

def k_means(points: np.ndarray, k: int, max_iterations: int = 100) -> np.ndarray:
    """
    Performs k-Means-Clustering.

    Parameters
    ----------
    points : np.ndarray
        (nr x dim) array of points to cluster
    k: int
        number of cluster centers
    max_iterations: int
        maximum number of k-Means-iterations

    Returns
    -------
    np.ndarray
        (k x dim) array with the cluster centers
    """
    cluster_idx = np.random.choice(len(points), k, replace=False)
    centroids = {}
    for i in range(len(cluster_idx)):
        centroids[i] = points[cluster_idx[i]]
    classification = {}
    for c in range(len(centroids)):
        classification[c] = []
    for ite in range(max_iterations):
        for i in range(len(points)):
            distances = []
            for c in range(len(centroids)):
                distance = np.linalg.norm(centroids[c]-points[i])
                distances.append(distance)
            feature_label = np.argmin(distances)
            classification[feature_label].append(points[i]) 
        for label_feat in classification:
            centroids[label_feat]= (np.mean(classification[label_feat],axis=0))
        
    return centroids 

def cluster_plotter(data,features,centroids):
    """
    Generates the clustering of datapoints

    Args:
        data (tensor) : dataset to be clustered
        features (tensor): train features
        centroids (tensor): trained centroids
    """
    num_images = 10
    plt.figure(figsize=(math.ceil(num_images * 0.7), math.ceil(num_images * 0.75)))
    for c in range(len(centroids)):
        distances = np.linalg.norm(features - centroids[c], axis=1)
        index = 0
        for i, d in sorted(enumerate(distances), key=lambda t: t[1])[:num_images]:
            plt.subplot(len(centroids), num_images, c * num_images + index + 1)
            plt.axis("off")
            plt.imshow(data[i], cmap="Greys")
            plt.title("{:.2f}".format(d))
            index += 1
    plt.show()

if __name__ == "__main__":
    checkpoint_path = "./models/best_model.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cell_size,block_size,bin_size,x_test,y_test,x_train,y_train,check_point,cntr = load_checkpoint(checkpoint_path)
    print(cell_size,block_size,bin_size,cntr)
    x_test_features = compute_hog(cell_size,block_size,bin_size,x_test)
    x_train_features = compute_hog(cell_size,block_size,bin_size,x_train)
    centroids = k_means(x_train_features,10)
    cluster_plotter(x_test,x_test_features,centroids)