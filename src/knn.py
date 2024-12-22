import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def minkowski_distance(self, x1, x2, p=1):
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

    def predict(self, X, dist_metric="Euclidean"):
        X = np.array(X)
        y_pred = [self._predict(x, dist_metric) for x in X]
        return np.array(y_pred)

    def _predict(self, x, dist_metric):
        x = np.array(x)
        # Compute distances between x and all examples in the training set
        if dist_metric == "Euclidean":
            distances = [self.euclidean_distance(x, X_train) for X_train in self.X_train]
        elif dist_metric == "Manhattan":
            distances = [self.manhattan_distance(x, X_train) for X_train in self.X_train]
        elif dist_metric == "Minkowski":
            distances = [self.minkowski_distance(x, X_train) for X_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
