# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import statsmodels.api as sm


class KDE(object):

    def __init__(self, data, weights=None, bw=None):
        """
        Setup and fit a kernel density estimator to data.

        Parameters
        ----------
        data : ndarray
            Data samples from which to build the KDE.
            The first dimension of the array defines the number of kernels (samples).
            The second dimension defines the number of features per sample.
        weights : ndarray
            Weights for each sample. Array should be of shape (n_samples, 1).
            The summation of all weights should equal to 1.
            If None is used, all samples are weighted equally.
        bw : ndarray
            The bandwidth vector to use. Array should be of shape (n_features, 1)
            If None is used, kernel sizes are automatically determined.

        """
        self.points = data
        self.n_kernels, self.n_features = self.points.shape
        self.weights = weights if weights is not None else np.repeat(1. / self.n_kernels, self.n_kernels)

        if bw is None:
            # first attempt to find the optimal BW using Hall's method
            # TODO: replace Hall's method with (newer) Improved Sheather-Jones method
            #       see DOI 10.1080/10485250903194003
            try:
                with np.errstate(all='raise'):
                    bw = self._ksizeHall(self.points)
            except:
                bw = np.array([np.nan])
            # do multivariate Rule of Thumb if Hall's BW is unusable
            if np.isnan(bw).any() or np.isinf(bw).any():
                bw = self._ksizeROT(self.points)

        # replace any zero widths with a small value
        self.bw = bw + (bw == 0.) * 0.001

        var_vector = ''.join(['c'] * self.n_features)
        self._kde = sm.nonparametric.KDEMultivariate(self.points, var_vector, bw=self.bw)

    def sample(self, n_samples):
        """
        Draw random samples from the estimator.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        ndarray
            A numpy matrix containing samples, of size (n_samples, n_features)
        """
        bw = np.tile(self.bw, (self.n_kernels, 1))
        points = np.zeros((n_samples, self.n_features))
        randnums = np.random.normal(size=(n_samples, self.n_features))

        # weights and thresholds to determine which kernel to sample from
        w = np.cumsum(self.weights)
        w /= np.amax(w)  # kernel weights represented as normalized cumsum
        t = list(np.sort(np.random.uniform(size=(n_samples,))).tolist())
        t.append(1.)     # final threshold value signals sampling is done

        ii = 1
        for i in range(self.n_kernels):
            # if kernel weight is less than threshold, go to next kernel
            # otherwise, continue sampling from current kernel
            while w[i] > t[ii]:
                points[ii, :] = self.points[i, :] + (bw[i, :] * randnums[ii, :])
                ii += 1
        # verify samples are correctly shaped before returning samples
        assert(points.shape[0] == n_samples)
        assert(points.shape[1] == self.points.shape[1])
        return points

    def predict(self, data):
        """
        Predict probability estimate for samples.

        Parameters
        ----------
        data : ndarray
            Data is a numpy array of dimensions (n_samples, n_features).
            The number of features in the data must be the same as the 
            number of features in the data used to fit the estimator.

        Returns
        -------
        ndarray
            A 1D numpy array containing the probabilities for each sample.

        """
        return self._kde.pdf(data)

    def entropy(self, data=None):
        """
        Calculate a resubstitute entropy estimate using the mean log-likelihood

        Parameters
        ----------
        data : ndarray
            Data samples to use for estimation, of shape (n_samples, n_features)
            If None is used, the samples and weights used to fit the KDE are used.

        Returns
        -------
        float
            Entropy estimate based on the mean log-likelihood

        """
        if data is not None:
            probs = self.predict(data)
            if np.any(probs[probs == 0.]):
                return -np.inf
            else:
                return -np.mean(np.log(probs))
        else:
            probs = self.predict(self.points)
            if np.any(self.weights[probs <= 0.]):
                return -np.inf
            else:
                probs[probs == 0.] = 1.
                return -np.dot(np.log(probs), np.transpose(self.weights))

    @staticmethod
    def _ksizeROT(X):
        """
        Find optimal kernel bandwidth using the Multivariate 'Rule of Thumb' technique.

        Parameters
        ----------
        X : ndarray
            Numpy array containing data samples, of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Numpy array containing optimal bandwidth for each dimension, of shape (n_features,)
        """
        X = np.transpose(X)

        noIQR = 0
        dim = X.shape[0]
        N = X.shape[1]

        prop = 1.0
        sig = np.std(X, axis=1)
        if noIQR:
            h = prop * sig * np.power(N, (-1 / (4 + dim)))
        else:
            iqrSig = .7413 * np.transpose(stats.iqr(np.transpose(X)))
            if np.amax(iqrSig) == 0:
                iqrSig = sig
            h = prop * np.minimum(sig, iqrSig) * np.power(float(N), (-1 / (4 + dim)))
        return h
    
    @staticmethod
    def _ksizeHall(X):
        """
        Find optimal kernel bandwidth using the "plug-in" method described by Hall et. al.

        The method will fail to find valid bandwidths when the variance between samples is zero.
        The caller needs to handle these scenarios.
        Method details can be found in DOI: 10.2307/2337251

        Parameters
        ----------
        X : ndarray
            Numpy array containing data samples, of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Numpy array containing optimal bandwidth for each dimension, of shape (n_features,)
        """
        X = np.transpose(X)
    
        N1, N2 = X.shape
        sig = np.std(X, axis=1)
        lamS = .7413 * np.transpose(stats.iqr(np.transpose(X)))
        if np.amax(lamS) == 0:
            lamS = sig

        BW = 1.0592 * lamS * np.power(float(N2), -1 / (4 + N1))
        BW = np.tile(BW, (1, N2))
    
        t = np.transpose(X[:, :, None], (0, 2, 1))
        dX = np.tile(t, (1, N2, 1))
    
        for i in range(N2):
            dX[:, :, i] = np.divide(dX[:, :, i] - X,
                                    BW)
        for i in range(N2):
            dX[:, i, i] = 2e22
        dX = np.reshape(dX, (N1, N2*N2))
    
        def h_findI2(n, dXa, alpha):
            t = np.exp(-0.5*np.sum(np.power(dXa, 2), axis=0))
            t = (np.power(dXa, 2) - 1) * (1/np.sqrt(2*np.pi)) * np.tile(t, (dXa.shape[0], 1))
            s = np.sum(t, axis=1)
            return np.divide(s, n*(n-1)*np.power(alpha, 5))
    
        def h_findI3(n, dXb, beta):
            t = np.exp(-0.5*np.sum(np.power(dXb, 2), axis=0))
            t = (np.power(dXb, 3) - (3*dXb)) * (1/np.sqrt(2*np.pi)) * np.tile(t, (dXb.shape[0], 1))
            s = np.sum(t, axis=1)
            return -np.divide(s, n*(n-1) * np.power(beta, 7))
    
        I2 = h_findI2(N2, dX, BW[:, 1])
        I3 = h_findI3(N2, dX, BW[:, 1])
    
        RK, mu2, mu4 = 0.282095, 1.000000, 3.000000
    
        J1 = (RK / mu2**2) * (1./I2)
        J2 = (mu4 * I3) / (20 * mu2) * (1./I2)
        h = np.power((J1/N2).astype(dtype=np.complex), 1.0/5) + \
            (J2 * np.power((J1/N2).astype(dtype=np.complex), 3.0/5))
        h = h.real.astype(dtype=np.float64)
    
        return np.transpose(h)
