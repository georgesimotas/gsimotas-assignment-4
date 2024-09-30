import numpy as np
from PIL import Image as im
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
import random

class KMeans:
    def __init__(self, data, k=3, init_method="random", max_iterations=100):
        self.data = data
        self.k = k
        self.init_method = init_method
        self.max_iterations = max_iterations
        self.centroids = None
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []
        self.has_converged = False

    def snap(self, centers):
        TEMPFILE = "temp.png"
        fig, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
        ax.scatter(centers[:, 0], centers[:, 1], c='r')
        fig.savefig(TEMPFILE)
        plt.close(fig)
        self.snaps.append(im.fromarray(np.asarray(im.open(TEMPFILE))))

    def initialize(self):
        if self.init_method == "random":
            return self.random_initialization(self.data)
        elif self.init_method == "farthest_first":
            return self.farthest_first_initialization(self.data)
        elif self.init_method == "kmeans++":
            return self.kmeans_plus_plus_initialization(self.data)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

    def make_clusters(self, centers):
        for i in range(len(self.assignment)):
            distances = [self.dist(centers[j], self.data[i]) for j in range(self.k)]
            self.assignment[i] = np.argmin(distances)

    def compute_centers(self):
        centers = []
        for i in range(self.k):
            cluster = [self.data[j] for j in range(len(self.assignment)) if self.assignment[j] == i]
            if cluster:
                centers.append(np.mean(np.array(cluster), axis=0))
            else:
                centers.append(self.centroids[i])  # Keep previous centroid if no points are assigned
        return np.array(centers)

    def dist(self, x, y):
        return np.linalg.norm(x - y)

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers, tol=1e-4):
        """
        Check if the new centers are different from the old ones, within a tolerance.
        """
        return np.any(np.linalg.norm(new_centers - centers, axis=1) > tol)

    def fit(self, step_by_step=False):
        """
        Fits KMeans to the data X step-by-step if required.
        """
        if step_by_step:
            # Perform one step and return current centroids and assignments
            if self.centroids is None:
                self.centroids = self.initialize()
            
            self.make_clusters(self.centroids)
            new_centroids = self.compute_centers()
            
            if np.all(new_centroids == self.centroids):
                return self.centroids, self.assignment  # Converged, no changes
            
            self.centroids = new_centroids  # Update centroids for the next step
            return self.centroids, self.assignment  # Return current centroids and assignments
        
        # Full KMeans execution (without step-by-step)
        for _ in range(self.max_iterations):
            self.make_clusters(self.centroids)
            new_centroids = self.compute_centers()

            if np.all(new_centroids == self.centroids):
                break  # Converged

            self.centroids = new_centroids
            self.unassign()

        return self.centroids, self.assignment


    def random_initialization(self, X):
        """
        Randomly choose k data points as initial centroids.
        """
        return X[np.random.choice(len(X), self.k, replace=False)]

    # Additional initialization methods
    def farthest_first_initialization(self, X):
        centroids = [X[np.random.randint(len(X))]]
        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self, X):
        centroids = [X[np.random.randint(len(X))]]
        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])
            probabilities = distances / distances.sum()
            next_centroid = X[np.random.choice(len(X), p=probabilities)]
            centroids.append(next_centroid)
        return np.array(centroids)

if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    centers = [[0, 0], [2, 2], [-3, 2], [2, -4]]
    X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=0)

    kmeans = KMeans(X, k=4, init_method="kmeans++")
    centroids, clusters = kmeans.fit()

    images = kmeans.snaps
    images[0].save(
        'kmeans.gif',
        optimize=False,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=500
    )
