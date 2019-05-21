
import math
from data_utils import logger
from collections import Iterable
from matlab.wrapper import MatlabWorkspace, AKDE
import numpy as np


class WebsiteFingerprintModeler(object):

    def __init__(self, data, sample_size=5000):
        """
        Initialize parameters of the fingerprint modeler
        :param X: instances as numpy matrix
        :param Y: labels
        :param sample_size: number of samples to use for monte-carlo estimation
        """
        self.data = data
        self.sample_size = sample_size

    def _make_kde(self, features, site=None, workspace=None):
        """
        Produce AKDE for a single feature or single feature for a particular site.
        :param features: index of feature(s) of which to model a multi/uni-variate AKDE
        :param site: (optional) model features only for the given website
        """
        if not isinstance(features, Iterable):
            features = [features]

        # build X for features
        X = None
        for feature in features:

            # get feature vector
            if site is not None:    # pdf(f|c)
                X_f = self.data.get_site(site, feature)
            else:       # pdf(f)
                X_f = self.data.get_feature(feature)
            X_f = np.reshape(X_f, (X_f.shape[0], 1))

            # extend X w/ feature vector if it has been initialized
            # otherwise, initalize X using the current feature vector
            if X is None:
                X = X_f
            else:
                np.hstack((X, X_f))

        # fit KDE on X
        return AKDE(workspace).fit(X)

    def _sample(self, mkdes, web_priors, sample_size):
        """
        Select samples for monte-carlo evaluation.
        Sampling is done on each website KDE.
        The number of samples drawn from each website is determined by prior.
        :param mkdes: list of site KDEs from which to sample
        :param web_priors: list of website priors
        :param sample_size: number of samples to generate
        """
        samples = []
        for site_mkdes in mkdes:
            group_samples = []
            for site, mkde in zip(self.data.sites, site_mkdes):

                # n = k * pr(c[i]) -- number of samples per site
                num = int(sample_size * web_priors[site])

                if num > 0:
                    # sample from pdf(f|c[i])
                    x = mkde.sample(num)
                    group_samples.extend(x)
            samples.append(group_samples)

        return samples

    def information_leakage(self, clusters):
        """
        Evaluate the information leakage.
        Computes marginal KDEs for features given a sites using the awkde library.
        Conditional entropy estimated via monte-carlo integration.
        :param features: index of feature(s) whose joint leakage should be produced
        """
        # catch unhandled errors
        try:
            # convert one feature to singular list for comparability
            if not isinstance(clusters, Iterable):
                clusters = [clusters]
            if not isinstance(clusters[0], Iterable):
                clusters = [clusters]

            logger.debug("Measuring leakage for {}".format(clusters))

            # create a workspace
            workspace = MatlabWorkspace()

            # create pdf for sampling and probability calculations
            cluster_mkdes = [[self._make_kde(features, site, workspace) for site in self.data.sites] for features in clusters]

            # Shannon Entropy func: -p(x)*log2(p(x))
            h = lambda x: -x * math.log(x, 2)

            # H(C) -- compute website entropy
            website_priors = [1/float(len(self.data.sites)) for _ in self.data.sites]
            H_C = sum([h(prior) for prior in website_priors if prior > 0])

            # performing sampling for monte-carlo evaluation of H(C|f)
            cluster_samples = self._sample(cluster_mkdes, website_priors, self.sample_size)

            # get probabilities of samples from each feature-website density distribution (for each cluster)
            cluster_prob_set = [[mkde.predict(samples) for mkde in site_mkdes]
                                for site_mkdes, samples in zip(cluster_mkdes, cluster_samples)]

            # independence is assumed between clusters
            # get final joint probabilities by multiplying sample probs of clusters together
            prob_set = []
            for i in range(len(cluster_prob_set[0])):
                prob = 1
                for j in range(len(cluster_prob_set)):
                    prob *= cluster_prob_set[j][i]
                prob_set.append(prob)

            # teardown matlab session
            del workspace

            # transpose array so that first index represents samples, and the second index represents features
            prob_set = np.array(prob_set).transpose((1, 0))

            # weight by website priors
            prob_temp = [[prob*prior for prob, prior in zip(prob_inst, website_priors)]
                         for prob_inst in prob_set]

            # normalize probabilities?
            prob_indiv = [[prob / sum(prob_inst) for prob in prob_inst]
                          for prob_inst in prob_temp]

            # check for calculation error?
            for prob_inst in prob_indiv:
                assert(sum(prob_inst) > 0.99)

            # compute entropy for instances
            entropies = [sum([h(prob) for prob in prob_inst if prob > 0])
                         for prob_inst in prob_indiv]

            # H(C|f) -- compute conditional entropy via monte-carlo from sample probabilities
            H_CF = sum(entropies)/len(entropies)

            # I(C;f) = H(C) - H(C|f) -- compute information leakage
            leakage = H_C - H_CF

            # debug output
            logger.debug("{l} = {c} - {cf}"
                         .format(l=leakage, c=H_C, cf=H_CF))

            return leakage

        except not KeyboardInterrupt:
            # in cases where there is an unknown error, save leakage as N/A
            # ignore these features when computing combined leakage
            logger.exception("Exception when estimating leakage for {}.".format(clusters))
            return None

    def __call__(self, features):
        return self.information_leakage(features)

