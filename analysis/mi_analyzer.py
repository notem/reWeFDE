# -*- coding: utf-8 -*-
import numpy as np
import os
from itertools import repeat, combinations_with_replacement
from data_utils import logger
#from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from kde_wrapper import KDE
from collections import Iterable
from scipy import stats


class MutualInformationAnalyzer(object):

    def __init__(self, data, pool=None):
        """
        Instantiate a MutualInformationAnalyzer object.

        Parameters
        ----------
        data : WebsiteData
            Website trace data.
        pool : ProcessPool
            An initialized pathos ProcessPool if multiprocessing should be used.
            Do not do multiprocessing if None.

        Returns
        -------
        MutualInformationAnalyzer

        """
        self.data = data
        self.nmi_threshold = 0.99
        self._pool = pool
        self._mi_cache = dict()
        self._nmi_cache = []

    def _estimate_entropy(self, features, site):
        """
        Produce an entropy estimate using the mean log-likelihood.
        To estimate entropy, first an AKDE model is fit to the data for the features & sites.
        The mean log-likelyhood is then computed by dividing total log-likelihood by the number of data instances.
        This entropy estimate assumes that all data instances have the same likelihood of occurring (ie. equal weights).

        Parameters
        ----------
        features : list
            List of feature numbers to represent in the data.
        sites : list
            List of site numbers to represent in the data.

        Returns
        -------
        float
            Entropy estimate for the given data.
        """
        # wrap arguments into lists if singular items were given
        if not isinstance(features, Iterable):
            features = [features]

        X = None   # data array used to fit KDE
        bw = []    # kernel bw for each feature
        for feature in features:

            # get feature vector
            if site is not None:    # pdf(f|c)
                X_f = self.data.get_site(site, feature)
            else:       # pdf(f)
                X_f = self.data.get_feature(feature)
            X_f = np.reshape(X_f, (X_f.shape[0], 1))

            # compute simple bandwidth estimate for feature
            # using Rule of Thumb or Hall's bw estimation seems to result in invalid entropy estimates
            bw_f = 0.9 * min(np.std(X_f), stats.iqr(X_f)/1.34) * X_f.shape[0]**(-0.2)
            bw.append(bw_f if bw_f > 0 else 0.1)

            # extend X w/ feature vector if it has been initialized
            # otherwise, initalize X using the current feature vector
            if X is None:
                X = X_f
            else:
                X = np.hstack((X, X_f))

        # fit a kde
        kde = KDE(X, bw=np.array(bw))

        # return the negative of the kde score (ie. total log likelihood) by the number of instances in data
        # this computation assumes that all data instances occur with equal weight
        return kde.entropy()

    def _avg_mi(self, feature_pair):
        """
        Calculate the average pairwise mutual information estimate.
        Computes an estimate of the average MI for a pair of features across all sites in the dataset.
        This is an approximation of the global MI value, and is used in the original WeFDE implementation.
        This trick substantially reduced computation time (x5).

        Parameters
        ----------
        feature_pair : tuple
            2-tuple of feature pair to process

        Returns
        -------
        float
            Averaged MI value
        """
        c, r = feature_pair

        # calculate the pairwise MI for data for each site
        mi_list = []
        for site in self.data.sites:
            # estimate entropy for each distribution
            c_entropy = self._estimate_entropy([c], site)
            r_entropy = self._estimate_entropy([r], site)
            cr_entropy = self._estimate_entropy([c, r], site)

            # calculate mutual information
            mi_estimate = c_entropy + r_entropy - cr_entropy
            mi_list.append(mi_estimate)

        # return the average of the MIs
        return sum(mi_list) / len(mi_list)

    def _estimate_nmi(self, feature_pair):
        """
        Estimate pairwise normalized mutual information value.

        Parameters
        ----------
        feature_pair : tuple
            2-tuple of feature pair to process

        Returns
        -------
        float
            Normalized MI value between 0.0 and 1.0.

        """
        c, r = feature_pair

        # measure MI for both single features
        # the max of these two values are used to normalize the joint MI
        # these values are saved in a separate internal cache
        mi_1 = self._mi_cache.get('{},{}'.format(c, c), None)
        mi_2 = self._mi_cache.get('{},{}'.format(r, r), None)
        if mi_1 is None:
            mi_1 = self._avg_mi((c, c))
            self._mi_cache['{},{}'.format(c, c)] = mi_1
        if mi_2 is None:
            mi_2 = self._avg_mi((r, r))
            self._mi_cache['{},{}'.format(r, r)] = mi_2

        # calculate entropies and mutual information of feature c and r
        mi = self._avg_mi(feature_pair)

        # calculate normalized mutual information
        return mi/max([mi_1, mi_2])

    def _check_redundancy(self, feature_pair):
        """
        Perform pairwise mutual information analysis to identify feature redundancy.

        Parameters
        ----------
        feature_pair : tuple
            2-tuple containing features to compare

        Returns
        -------
        tuple
            Tuple of three object: (redundancy, feature_pair, nmi_value)
        """
        feature1, feature2 = feature_pair

        # calculate the normalized mutual information
        nmi = self._estimate_nmi((feature1, feature2))
        logger.debug("| nmi({},{}) = {}".format(feature1, feature2, nmi))

        # prune if nmi is above threshold
        if nmi > self.nmi_threshold:
            logger.debug("Feature #{} is redundant w/ #{}".format(feature1, feature2))
            return True, feature_pair, nmi
        # feature is not redundant
        return False, feature_pair, nmi

    def prune(self, features, checkpoint=None, nmi_threshold=0.9, topn=100):
        """
        Reduce the feature-set to a list of top features which are non-redundant.

        Redundancy is identified by estimating the pair-wise mutual information of features.
        The algorithm will find up to a maximum of ``topn`` non-redundant features before ending.
        If the MIAnalyzer was instantiated with a ``pool``, NMI calculations will be performed in parallel.

        Parameters
        ----------
        features : list
            Array of features from which to prune redundant features.
            Features should be pre-sorted by importance with the most important feature being at index 0.
        checkpoint : str
            Path to plaintext file to store feature redundancy checkpoint information.
            Do not perform checkpointing if None is used.
        nmi_threshold : float
            Threshold value used to identify redundant features.
            Features with NMI values greater than the threshold value are pruned.
        topn : int
            Number of features to save when pruning is performed.

        Returns
        -------
        list
            Features list having variable length up to ``topn``.
        """
        # results of NMI calculations are saved in list internal to the analyzer
        # reduces the amount of computation required in any subsequent cluster calls
        self._nmi_cache, self._mi_cache = [], dict()

        self.nmi_threshold = nmi_threshold

        # feature lists
        cleaned_features = set()  # non-redundant
        pruned_features = set()   # redundant

        # if checkpointing, open file and read any previously processed features
        if checkpoint is not None:
            if os.path.exists(checkpoint):
                checkpoint_fi = open(checkpoint, 'r+')
                for line in checkpoint_fi:
                    try:
                        if line[0] == '+':
                            feature = int(line[1:].strip())
                            cleaned_features.add(feature)
                        elif line[0] == '-':
                            feature = int(line[1:].strip())
                            pruned_features.add(feature)
                        if line[0] == '=':
                            a, b, c = line[1:].split(',')
                            self._nmi_cache.append(((int(a), int(b)), float(c)))
                    except:
                        pass
                features = list(filter(lambda f: f not in cleaned_features and f not in pruned_features, features))
                checkpoint_fi.close()

            # re-open checkpoint for appending
            checkpoint = open(checkpoint, 'a+')

        # continue to process features until either there are no features left to process
        # or the topN features have been selected
        while features and len(cleaned_features) < topn:

            # the next most important feature
            current_feature = features.pop(0)
            logger.debug("MI analysis on feature #{}".format(current_feature))

            # for all top features, measure pair-wise mutual information to check for redundancy
            feature_pairs = zip(repeat(current_feature), cleaned_features)
            if self._pool is None or len(cleaned_features) < 2:
                results = map(self._check_redundancy, feature_pairs)
            else:   # parallel, unordered
                results = self._pool.uimap(self._check_redundancy, feature_pairs)

            # break upon first occurrence of redundancy
            is_redundant = False
            for res in results:

                # unzip results
                is_redundant, feature_pair, nmi = res

                # save feature pair with nmi in cache
                self._nmi_cache.append((feature_pair, nmi))
                if checkpoint is not None:
                    checkpoint.write('={},{},{}\n'.format(feature_pair[0], feature_pair[1], nmi))
                    checkpoint.flush()

                # break loop
                if is_redundant:
                    # if the analyzer is using a process pool
                    # terminate processes and restart the pool
                    if self._pool is not None:
                        self._pool.terminate()
                        self._pool.join()
                        self._pool.restart()
                    break

            # if the current feature does not appear to be redundant with any
            # other top features, add current feature to top features list
            if not is_redundant:
                cleaned_features.add(current_feature)
                logger.info("Progress: {}/{}".format(len(cleaned_features), min(topn, len(features))))
                if checkpoint is not None:
                    checkpoint.write('+{}\n'.format(current_feature))
                    checkpoint.flush()
            else:
                pruned_features.add(current_feature)
                if checkpoint is not None:
                    checkpoint.write('-{}\n'.format(current_feature))
                    checkpoint.flush()

        if checkpoint is not None:
            checkpoint.close()

        # return both non-redundant and redundant features
        # which feature was redundant with which is however not saved
        return list(cleaned_features), list(pruned_features)

    def cluster(self, features, checkpoint=None, min_samples=1, min_cluster_size=3):
        """
        Find clusters in provided features.

        Use DBSCAN algorithm to cluster topN features based upon their pairwise mutual information.
        First fill an NxN matrix with NMI feature pair values.
        NMI values may be retrieved from the MIAnalyzer's internal cache or by doing computations anew.
        The DBSCAN model is then fit to this distances grid, and the identified clusters are returned.

        Parameters
        ----------
        features : list
            A list of features to cluster
        checkpoint : str
            Path to plaintext file to store feature redundancy checkpoint information.
            Do not perform checkpointing if None is used.
        min_samples : int
            The min_samples parameter to use for the HDBSCAN algorithm.
            The number of samples in a neighbourhood for a point to be considered a core point.

        min_cluster_size : int
            The min_cluster_size parameter to use for the HDBSCAN algorithm.
            The minimum size of clusters; single linkage splits that contain fewer points than this will be considered points “falling out” of a cluster rather than a cluster splitting into two new clusters.

        Returns
        -------
        list
            Nested lists where each list contains the cluster's features.
            Features that do not fall into a cluster are given their own cluster (ie. singular list).
        """
        # compute pairwise MI for all topN features
        X = np.empty(shape=(len(features), len(features)), dtype=float)  # distance matrix
        pairs = list(combinations_with_replacement(features, 2))         # all possible combinations

        # if checkpointing, read NMI calculations and save to cache
        if checkpoint is not None:
            if os.path.exists(checkpoint):
                chk_fi = open(checkpoint, 'r+')
                for line in chk_fi:
                    try:
                        if line[0] == '=':
                            a, b, c = line[1:].split(',')
                            self._nmi_cache.append(((int(a), int(b)), float(c)))
                    except:
                        pass
                chk_fi.close()
            # re-open checkpoint for appending
            chk_fi = open(checkpoint, 'a+')

        if self._nmi_cache:
            # ignore unselected features in cache
            cache = [(pair, nmi) for pair, nmi in self._nmi_cache if pair[0] in features and pair[1] in features]
            # add each cached nmi to the distance matrix
            for cached_pair, nmi in cache:
                # remove cached_pair from pairs
                pairs = list(filter(lambda pair: (pair[0] != cached_pair[0] and pair[1] != cached_pair[1]) and
                                                 (pair[0] != cached_pair[1] and pair[1] != cached_pair[0]), pairs))
                # add cached nmi to matrix
                i, j = features.index(cached_pair[0]), features.index(cached_pair[1])
                X[i][j] = 1 - nmi
                X[j][i] = 1 - nmi

        if len(pairs) > 0:
            # map pairs to nmi
            if self._pool is None:
                results = map(self._estimate_nmi, pairs)
            else:
                results = self._pool.imap(self._estimate_nmi, pairs)
                self._pool.close()

            # fill matrix with pair nmi values
            count = 0
            for pair, nmi in zip(pairs, results):

                # print progress updates
                count += 1
                if count-1 % (len(pairs)*0.05) == 0:
                    logger.info("Progress: {}/{}".format(count, len(pairs)))

                fidx1, fidx2 = pair
                i, j = features.index(fidx1), features.index(fidx2)
                X[i][j] = 1 - nmi
                X[j][i] = 1 - nmi

                if checkpoint is not None:
                    chk_fi.write('={},{},{}\n'.format(fidx1, fidx2, nmi))
                    chk_fi.flush()

            # restart pool if multiprocessing
            if self._pool is not None:
                self._pool.join()
                self._pool.restart()

        # verify that all values are filled
        assert not np.any(X[X == np.nan])

        # use DBSCAN to cluster our data
        labels = HDBSCAN(metric='precomputed',
                         min_samples=min_samples,
                         min_cluster_size=min_cluster_size).fit_predict(X)
        logger.debug("Found {} clusters.".format(set(labels)))

        # organize the topN features into sub-lists where
        # each sub-list contains all features in a cluster
        clusters = []
        for label in range(min(labels), max(labels)+1):
            if label >= 0:
                cluster = [features[i] for i, la in enumerate(labels) if la == label]
                clusters.append(cluster)
            else:
                # treat features that do not cluster (ie. noise) each as their own independent cluster
                noise = [[features[i]] for i, la in enumerate(labels) if la == label]
                clusters.extend(noise)

        logger.debug("Clusters: {}".format(labels))
        return clusters, X
