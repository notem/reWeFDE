
import math
import numpy as np

from awkde.awkde import GaussianKDE
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional


class WebsiteData(object):
    """
    Object-wrapper to conveniently manage dataset
    """

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


class ConditionalKDE(object):
    """
    Implementation of a conditional probability density
        function using awkde GaussianKDE as the estimators
    Conditional KDE defined as:
        f(y|x) = f(x,y)/f(x)
    """

    def __init__(self, dep, indep):
        dep = np.reshape(dep, (dep.shape[0], 1))
        indep = np.reshape(indep, (indep.shape[0], 1))
        joint = np.hstack((dep, indep))

        self.joint = GaussianKDE()
        self.indep = GaussianKDE()

        self.joint.fit(joint)
        self.indep.fit(indep)

    def pdf(self, dep, indep):
        # dep = np.array(dep)
        # indep = np.array(indep)
        #
        joint = np.reshape(np.array([dep[0], indep[0]]), (1, 2))
        indep = np.reshape(np.array([indep]), (1, 1))

        xy = self.joint.predict(joint)
        x = self.indep.predict(indep)
        pred = np.divide(xy, x, out=np.zeros_like(xy), where=(x != 0))

        return pred


class WebsiteFingerprintModeler(object):

    def __init__(self, X, Y):
        """
        :param X: instances as numpy matrix
        :param Y: labels
        """
        self.data = WebsiteData(X, Y)

    def _model_individual(self, feature, site=None):
        """
        Produce KDE for a single feature or single feature for a particular site
        """
        if site:
            # pdf(f|c)
            X = self.data.get_site(site)[:, feature]
        else:
            # pdf(f)
            X = self.data.X[:, feature]
        X = np.reshape(X, (X.shape[0], 1))
        kde = GaussianKDE()
        kde.fit(X)
        return kde

    def _model_conditionals(self, feature):
        """
        Produce conditional pdfs for C|f
        """
        X = self.data.X[:, feature]
        Y = self.data.Y
        kde = ConditionalKDE(dep=Y, indep=X)
        return kde

    @staticmethod
    def _calculate_entropy(probs):
        """
        Compute the Shannon entropy given probabilities
        """
        # Shannon Entropy func: -p(x)*log2(p(x))
        e = lambda i: -probs[i] * math.log2(probs[i])
        sequence = [e(i) for i in range(len(probs)) if probs[i] > 0]
        return sum(sequence)

    def _sample(self, feature, web_priors, sample_size):
        """
        Select samples for monte-carlo evaluation
        """
        samples = []
        for i, site in enumerate(self.data.sites):

            # n = k * pr(c[i]) -- number of samples per site
            num = int(sample_size*web_priors[i])

            # distribution pdf(f|c[i])
            kde = self._model_individual(feature=feature, site=site)

            # sample from pdf(f|c[i])
            x = kde.sample(n_samples=num)
            samples.extend(x)

        return samples

    def individual_leakage(self, feature, sample_size=5000):
        """
        Evaluate the information leakage.
        """
        # create pdf for sampling and probability calculations
        skde = self._model_conditionals(feature)

        # H(C) -- compute website entropy
        website_priors = [1/len(self.data.sites) for _ in self.data.sites]
        H_C = self._calculate_entropy(website_priors)

        # H(C|f) -- compute conditional entropy via monte-carlo
        samples = self._sample(feature, website_priors, sample_size)
        H_CF = 0
        for i, sample in enumerate(samples):
            print("\tcomputing monte-carlo sample: {}".format(i), end="\r")
            probs = [skde.pdf([site], [sample]).tolist()[0] for site in self.data.sites]
            entropy = self._calculate_entropy(probs)
            H_CF += entropy
        H_CF /= sample_size

        # I = H(C) - H(C|f)
        leakage = H_C - H_CF

        # debug output
        print("\tH(C) = {}".format(H_CF))
        print("\tH(C|f) = {}".format(H_CF))
        print("\tI(C;f) = {}".format(leakage))
        return leakage

