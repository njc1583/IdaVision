import numpy as np

from sklearn.mixture import GaussianMixture

class GaussianModel:
    def __init__(self, data, n_components):
        self.data = data

        self.n_components = n_components
        
        self.model = GaussianMixture(n_components=n_components).fit(self.data)

        self.last_match = 0

    def getSize(self):
        return self.data.shape[0]

    def calculatePriorBackground(self):
        return 1 / self.getSize()

    def calculatePriorForeground(self, N_f):
        return 1 / (self.last_match + N_f)

    def score_samples(self, X):
        return self.model.score_samples(X)

    def sample(self, n_samples):
        return self.model.sample(n_samples)

    def updateModel(self, X):
        self.data = np.append(self.data, X, axis=0)

        self.model = GaussianMixture(n_components=self.n_components).fit(self.data)

        self.last_match = 0

    def incrementLastMatch(self):
        self.last_match += 1