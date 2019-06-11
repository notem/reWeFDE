from KDEpy import FFTKDE, TreeKDE
import numpy as np
from scipy import stats


class AKDE(object):

    def __init__(self, data, weights=None):
        self._data = data
        self._weights = weights
        self._n_kernels, self._n_features = self._data.shape

        if self._n_features == 1:
            self._kde = FFTKDE(kernel='gaussian', bw='ISJ')
        else:
            bw = ksizeROT(data)
            self._kde = TreeKDE(kernel='gaussian', bw=bw)
        self._kde.fit(self._data, self._weights)

    def sample(self, n_samples):
        """

        """
        points, probs = self._kde.evaluate(n_samples)

        # verify correctness of points
        assert points.shape[0] == n_samples
        assert points.shape[1] == self._n_features

        return points

    def predict(self, data):
        """

        """
        return self._kde.evaluate(data)

    def entropy(self, data=None):
        """

        """
        if data is not None:
            probs = self.predict(data)
            if np.any(probs[probs == 0.]):
                return -np.inf
            else:
                return -np.mean(np.log(probs))
        else:
            probs = self.predict(self._data)
            weights = self._weights
            if weights is None:
                weights = np.repeat(1. / self._data.shape[0], self._data.shape[0])
            if np.any(weights[probs <= 0.]):
                return -np.inf
            else:
                probs[probs == 0.] = 1.
                return -np.dot(np.log(probs), np.transpose(weights))


def ksizeROT(X, type=0):
    """

    :param X:
    :param dim:
    :param N:
    :param type:
    :return:
    """
    noIQR = 0
    dim = X.shape[0]
    N = X.shape[1]

    Rg, Mg = .282095, 1
    Re, Me = .6, .199994
    Rl, Ml = .25, 1.994473

    if type == 1:  # Epanetchnikov
        prop = np.power(np.power(Re/Rg, dim) / np.power(Me/Mg, 2), (1 / (dim + 4)))
    elif type == 2:  # Laplacian
        prop = np.power(np.power(Rl/Rg, dim) / np.power(Ml/Mg, 2), (1 / (dim + 4)))
    else:   # Gaussian
        prop = 1.0

    sig = np.std(X, axis=0)
    if noIQR:
        h = prop * sig * np.power(N, (-1 / (4 + dim)))
    else:
        iqrSig = .7413 * np.transpose(stats.iqr(np.transpose(X)))
        if np.amax(iqrSig) == 0:
            iqrSig = sig
        h = prop * np.minimum(sig, iqrSig) * np.power(N, (-1 / (4 + dim)))
    return h