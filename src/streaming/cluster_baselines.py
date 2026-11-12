"""
Baseline clusterers for the clustering ablation (CLUSTER_METHOD env var).

They expose what training/inference use from the DP-GMM model:
predict, score_samples, weights_, n_components. K-Means has no likelihood,
so score_samples returns the negative squared distance to the nearest centre.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class KMeansModel:
    def __init__(self, n_clusters, seed):
        self.km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        self.n_components = n_clusters

    def fit(self, Z):
        labels = self.km.fit_predict(Z)
        self.weights_ = np.bincount(labels, minlength=self.n_components) / len(Z)
        self.means_ = self.km.cluster_centers_
        return self

    def predict(self, Z):
        return self.km.predict(Z)

    def score_samples(self, Z):
        return -self.km.transform(Z).min(axis=1) ** 2


def fit_baseline(method, k, cov_type, seed, Z):
    if method == 'kmeans':
        model = KMeansModel(k, seed).fit(Z)
    elif method == 'gmm':
        model = GaussianMixture(n_components=k, covariance_type=cov_type,
                                n_init=10, random_state=seed).fit(Z)
    else:
        raise ValueError(f'unknown CLUSTER_METHOD {method}')
    return model
