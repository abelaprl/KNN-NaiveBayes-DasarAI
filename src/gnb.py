import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes, features))
        self.var = np.zeros((n_classes, features))
        self.priors = np.zeros(n_classes)

        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.mean[idx, :] = np.mean(X_cls, axis=0)
            self.var[idx, :] = np.var(X_cls, axis=0)
            self.priors[idx] = X_cls.shape[0] / n_samples

    def _gaussian_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx] + 1e-9
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _posterior(self, x):
        posteriors = []
        for idx, cls in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self._gaussian_likelihood(idx, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = X.astype(float)
        y_pred = [self._posterior(x) for x in X]
        return np.array(y_pred)

    def get_params(self, deep=True):
        return {}