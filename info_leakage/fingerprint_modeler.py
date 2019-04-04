
import math
import numpy as np

from statsmodels.api.nonparametric import KDEMultivariate
from statsmodels.api.nonparametric import KDEMultivariateConditional


class WebsiteData(object):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.features = range(X.shape[1])
        self.sites = range(max(np.unique(self.Y)))

    def get_site(self, label):
        """
        Get X for given site
        """
        f = [True if y == label else False for y in self.Y]
        return self.X[f, :]


class WebsiteFingerprintModeler(object):

    def __init__(self, X, Y):
        """
        :param X: instances as numpy matrix
        :param Y: labels
        """
        self.data = WebsiteData(X, Y)

    def _model_individual(self, feature, site=None):
        """
        produce KDE for a single feature or single feature for a particular site
        """
        if site:
            X = self.data.get_site(site)[:, feature]
        else:
            X = self.data.X[:, feature]
        X = X.flatten()
        var_type = 'c'
        kde = KDEMultivariate(data=X,
                              var_type=var_type,
                              bw='normal_reference')
        return kde

    def _model_conditionals(self, feature):
        """
        Produce conditional pdfs for C|f and f|c
        """
        X = self.data.X[:, feature]
        Y = self.data.Y

        # pdf(C|f)
        skde = KDEMultivariateConditional(endog=Y,   # dependent var
                                          exog=X,    # independent var
                                          dep_type="d",
                                          indep_type="c",
                                          bw='normal_reference')
        # pdf(f|c)
        fkde = KDEMultivariateConditional(endog=X,   # dependent var
                                          exog=Y,    # independent var
                                          dep_type="c",
                                          indep_type="d",
                                          bw='normal_reference')
        return skde, fkde

    @staticmethod
    def _calculate_entropy(probs):
        """
        Compute the Shannon entropy given probabilities
        """
        # Shannon Entropy func: -p(x)*log2(p(x))
        e = lambda i: -probs[i] * math.log2(probs[i])
        sequence = [e(i) for i in range(len(probs)) if probs[i] != 0]
        return sum(sequence)

    def _sample(self, feature_kde, web_priors, sample_size):
        """
        Select samples for monte-carlo evaluation
        """
        samples = []
        for i, site in enumerate(self.data.sites):
            # n = k * pr(c[i]) -- number of samples per site
            num = sample_size*web_priors[i]

            # sample from pdf(f|c[i])
            for _ in range(num):
                # select random sample of feature given the current site
                x = 0  # TODO - sample from feature_kde?
                samples.append(x)

        return samples

    def individual_leakage(self, feature, sample_size=5000):
        """
        Evaluate the information leakage.
        """
        # create pdf for sampling and probability calculations
        skde, fkde = self._model_conditionals(feature)

        # H(C) -- compute website entropy
        website_priors = [1/len(self.data.sites) for _ in self.data.sites]
        H_C = self._calculate_entropy(website_priors)

        # H(C|f) -- compute conditional entropy via monte-carlo
        samples = self._sample(fkde, website_priors, sample_size)
        H_CF = 0
        for sample in samples:
            priors = [skde.pdf([site], [sample]) for site in self.data.sites]
            H_CF += self._calculate_entropy(priors)
        H_CF /= sample_size

        # I = H(C) - H(C|f)
        return H_C - H_CF

