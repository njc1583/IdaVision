import numpy as np

from sklearn.mixture import GaussianMixture

class GaussianModel:
    def __init__(self, data, n_components):
        self.data = data
        
        self.model = GaussianMixture(n_components=n_components).fit(self.data)

    def getSize(self):
        return self.data.shape[0]

    def calculatePriorBackground(self):
        return 1 / self.getSize()

    def calculatePriorForeground(self, t_k, N_f):
        return 1 / (t_k + N_f)

    def score_samples(self, X):
        return self.model.score_samples(X)