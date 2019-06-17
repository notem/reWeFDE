# -*- coding: utf-8 -*-
import math
from data_utils import logger
from collections import Iterable
from kde_wrapper import KDE
import numpy as np


class WebsiteFingerprintModeler(object):

    def __init__(self, data, pool=None):
        """
        Instantiate a fingerprint modeler.

        Parameters
        ----------
        data : WebsiteData
            Website trace data object

        """
        self.sample_size = 1000
        self.data = data
        self.website_priors = [1/float(len(self.data.sites)) for _ in self.data.sites]
        self._pool = pool

    def _make_kde(self, features, site=None):
        """
        Produce AKDE for a single feature or single feature for a particular site.

        Parameters
        ----------
        features : list
            Feature(s) of which to model a multi/uni-variate AKDE.
        site : int
            Model features only for the given website number.
            Model all sites if None.

        Returns
        -------
        KDE
            Fit KDE for the feature data

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
                X = np.hstack((X, X_f))

        # fit KDE on X
        return KDE(X)

    def _sample(self, mkdes, web_priors, sample_size):
        """
        Generate samples from site KDEs.

        Sampling is done on each website KDE.
        The number of samples drawn from each website is determined by prior.
        Selected samples are later used for monte-carlo evaluation.

        Parameters
        ----------
        mkdes : list
            list of site AKDEs from which to sample
        web_priors : list
            list of website priors
        sample_size : int
            number of samples to generate

        Returns
        -------
        list
            List of instance samples.
            The dimension of the samples depends on the number of features used to generate the KDEs.

        """
        samples = []
        for site, mkde in zip(self.data.sites, mkdes):

            # n = k * pr(c[i]) -- number of samples per site
            num = int(sample_size * web_priors[site])

            if num > 0:
                # sample from pdf(f|c[i])
                x = mkde.sample(num)
                samples.extend(x)

        return samples

    def _do_predictions(self, cluster):
        """
        Produce the site x prediction matrix for a cluster.

        Parameters
        ----------
        cluster : list
            Features to be modeled.

        Returns
        -------
        ndarray
            Numpy array of dimensions (n_sites, n_samples) containing
            the probability predictions for samples of each site.

        """
        mkdes = [self._make_kde(cluster, site) for site in self.data.sites]

        # performing sampling for monte-carlo evaluation of H(C|f)
        samples = self._sample(mkdes, self.website_priors, self.sample_size)

        # get probabilities of samples for each site
        probs = np.array([mkde.predict(samples) for mkde in mkdes])

        # sample predictions are often above 1.0 (related to the bw choice?)
        # this snippet of code is adapted from the original WeFDE implementation
        # seems to correctly adjust the predictions to values between 0.0 and 1.0
        with np.errstate(divide='ignore'):
            probs = np.log2(probs)
            probs[probs == np.nan] = -300
        probs = probs - np.amax(probs)
        probs = 2**probs
        assert not np.any(probs[probs > 1.0])

        return probs

    def information_leakage(self, clusters, sample_size=5000, joint_leakage=True):
        """
        Evaluate the information leakage for feature(s).

        Computes marginal KDEs for features given a sites using AKDEs.
        Conditional entropy is then estimated from the distributions via monte-carlo integration.
        The conditional entropy is then used to compute the leakage for the feature(s)

        Parameters
        ----------
        clusters : list
            A list of lists. Features is a list of clusters.
            Each cluster is a list containing the features in the cluster.
            A singular feature or cluster may be given as the parameter.
            In those instances, the data will be wrapped in additional lists to match the expected form.
        sample_size : int
            Count of total random feature samples to use for monte-carlo estimation.
        joint_leakage : bool
            Determines if the leakage of clusters should be measured jointly or individually.
            If True, the probability of samples for each cluster will be multiplied together before estimating entropy.
            Otherwise, the leakage for each cluster is measured.

        Returns
        -------
        list
            Estimated information leakage for the features/clusters.
            If ``joint_leakage`` is True, the list contains the leakage for the combined analysis.
            Otherwise, the list contains the leakages for each cluster,
            appearing in the same order as seen in ``clusters``.

        """
        # convert one feature to singular list for comparability
        if not isinstance(clusters, Iterable):
            clusters = [clusters]
        if not isinstance(clusters[0], Iterable):
            clusters = [clusters]

        self.sample_size = sample_size
        logger.debug("Measuring leakage for {}".format(clusters))

        # Shannon Entropy func: -p(x)*log2(p(x))
        h = lambda x: -x * math.log(x, 2)

        # H(C) -- compute website entropy
        H_C = sum([h(prior) for prior in self.website_priors if prior > 0])

        # map clusters to probability predictions for random samples
        # allows for KDE construction, sampling, and prediction to be done in parallel (if enabled)
        if self._pool is None:
            results = map(self._do_predictions, clusters)
        else:
            results = self._pool.imap(self._do_predictions, clusters)
            self._pool.close()

        # load the results
        cluster_probs = []
        for probs in results:
            cluster_probs.append(probs)
            # print progress updates
            count = len(cluster_probs)
            if count-1 % (len(clusters)*0.05) == 0:
                logger.info("Progress: {}/{}".format(count, len(clusters)))

        # restart pool if multiprocessing
        if self._pool is not None:
            self._pool.join()
            self._pool.restart()

        if joint_leakage:
            # multiply cluster probs to get joint probs for each sample
            cluster_probs = np.array(cluster_probs)
            prob_sets = [np.prod(cluster_probs, axis=0)]
        else:
            # measure leakages for each cluster independently
            prob_sets = cluster_probs

        # compute information leakage for each cluster (or combined cluster if joint)
        leakages = []
        for i, prob_set in enumerate(prob_sets):

            # weight the probability predictions by the website priors
            probs_weighted = []
            for site, probs in enumerate(prob_set):
                probs_weighted.append(probs * self.website_priors[site])
            probs_weighted = np.array(probs_weighted)

            # transpose array so that first index represents samples, second index represent site
            probs_weighted = np.transpose(probs_weighted)

            # normalize probabilities such that the per-site probs for each sample sums to one
            # (as should be expected for conditional probabilities)
            probs_norm = []
            for probs in probs_weighted:
                norm = probs / sum(probs) if sum(probs) > 0 else probs
                probs_norm.append(norm)

            # compute entropy for each sample
            entropies = []
            for probs in probs_norm:
                entropies.append(sum([h(prob) for prob in probs if prob > 0]))

            # H(C|f) -- estimate real entropy as average of all samples
            H_CF = sum(entropies)/len(entropies)

            # I(C;f) = H(C) - H(C|f) -- compute information leakage
            leakage = H_C - H_CF
            leakages.append(leakage)

            # debug output
            logger.debug("{cluster} {l} = {c} - {cf}"
                         .format(cluster=clusters[i], l=leakage, c=H_C, cf=H_CF))

        return leakages

    def __call__(self, features):
        return self.information_leakage(features)

