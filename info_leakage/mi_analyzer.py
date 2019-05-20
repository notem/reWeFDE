import numpy as np
from itertools import repeat, combinations_with_replacement
from data_utils import logger
from sklearn.cluster import DBSCAN
from matlab.wrapper import MatlabWorkspace


class MutualInformationAnalyzer(object):

    def __init__(self, data, leakage, nmi_threshold=0.9, topn=100, pool=None):
        self.data = data
        self.nmi_threshold = nmi_threshold
        self.topn = topn
        self.leakage = leakage
        self.pool = pool
        self._nmi_cache = []
        self._mi_cache = []

    def _nmi_helper(self, feature_pair, workspace):
        """
        Calculate the average pairwise mutual information value across sites.
        This is an approximation of the global MI value, and is used in the original WeFDE implementation.
        This trick substantially reduced computation time (x5).
        :param feature_pair: 2-tuple of feature pair to process
        :param workspace: matlab workspace which to run the process
        :return: average MI
        """
        c, r = feature_pair

        # calculate the pairwise MI for data for each site
        mi_list = []
        for site in self.data.sites:

            # setup data array
            X1 = self.data.get_feature(c, site)
            X1 = np.reshape(X1, (X1.shape[0], 1))
            X2 = self.data.get_feature(r, site)
            X2 = np.reshape(X2, (X2.shape[0], 1))
            X = np.hstack((X1, X2))

            mi_list.append(workspace.pairwise_mi(X))

        # return the average of the MIs
        return sum(mi_list)/len(mi_list)

    def _estimate_nmi(self, feature_pair, workspace=None):
        """
        Estimate pairwise normalized mutual information value.
        :param c: Feature number
        :param r: Feature number
        :return: normalized mutual information
        """
        c, r = feature_pair

        # create a matlab workspace if necessary
        del_workspace = False
        if workspace is None:
            del_workspace = True
            workspace = MatlabWorkspace()

        # measure MI for both single features
        # the max of these two values are used to normalize the joint MI
        # these values are saved in a separate internal cache
        mi_1 = None
        mi_2 = None
        for pair, mi in self._mi_cache:
            if pair[0] == c and pair[1] == c:
                mi_1 = mi
            elif pair[0] == r and pair[1] == r:
                mi_2 = mi
        if mi_1 is None:
            mi_1 = self._nmi_helper((c, c), workspace)
            self._mi_cache.append(((c, c), mi_1))
        if mi_2 is None:
            mi_2 = self._nmi_helper((r, r), workspace)
            self._mi_cache.append(((r, r), mi_2))

        # calculate entropies and mutual information of feature c and r
        mi = self._nmi_helper(feature_pair, workspace)

        # delete workspace if necessary
        if del_workspace:
            del workspace

        # calculate normalized mutual information
        return mi/max([mi_1, mi_2])

    def _check_redundancy(self, feature_pair):
        """
        Perform pairwise mutual information analysis to identify feature redundancy.
        :param feature_pair: 2-tuple containing features to compare
        :return: 3-tuple (redundancy, feature_pair, nmi_value)
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

    def prune(self):
        """
        Reduce the feature-set to a list of top N features which are non-redudant.
        Redundancy is identified by estimating the pair-wise mutual information of features.
        :return: list of features, length topn
        """
        # results of NMI calculations are saved in list internal to the analyzer
        # reduces the amount of computation required in any subsequent cluster calls
        self._nmi_cache, self._mi_cache = [], []

        # list of best features
        cleaned_features = []
        pruned_features = []

        # sort the list of features by their individual leakage
        # we will process these features in the order of their importance
        logger.debug("Sorting features by individual leakage.")
        tuples = zip(self.data.features, self.leakage)
        tuples = sorted(tuples, key=lambda x: (-x[1], x[0]))
        logger.debug("Top 50:\t {}".format(tuples[:50]))

        # continue to process features until either there are no features left to process
        # or the topN features have been selected
        while tuples and len(cleaned_features) < self.topn:

            # the next most important feature
            current_feature = tuples.pop(0)[0]
            logger.debug("MI analysis on feature #{}".format(current_feature))

            # for all top features, measure pair-wise mutual information to check for redundancy
            feature_pairs = zip(repeat(current_feature), cleaned_features)
            if self.pool is None or len(cleaned_features) < 2:
                results = map(self._check_redundancy, feature_pairs)
            else:   # parallel, unordered
                results = self.pool.uimap(self._check_redundancy, feature_pairs)

            # break upon first occurrence of redundancy
            is_redundant = False
            for res in results:

                # unzip results
                is_redundant, feature_pair, nmi = res

                # save feature pair with nmi in internal cache
                self._nmi_cache.append((feature_pair, nmi))

                # break loop
                if is_redundant:
                    # if the analyzer is using a process pool
                    # terminate processes and restart the pool
                    if self.pool is not None:
                        self.pool.terminate()
                        self.pool.join()
                        self.pool.restart()
                    break

            # if the current feature does not appear to be redundant with any
            # other top features, add current feature to top features list
            if not is_redundant:
                cleaned_features.append(current_feature)
                logger.info("Progress: {}/{}".format(len(cleaned_features), self.topn))
            else:
                pruned_features.append(current_feature)

        logger.info('Removed {} redundant features.'.format(len(pruned_features)))
        logger.debug("Pruned features: {}".format(pruned_features))
        logger.debug("Clean features: {}".format(cleaned_features))
        return cleaned_features

    def cluster(self, features, eps=0.4):
        """
        Use DBSCAN algorithm to cluster topN features based upon their pairwise mutual information.
        :param features: topN features
        :param eps: DBSCAN eps value
        :return: nested lists where each list contains the a cluster's features
        """

        # compute pairwise MI for all topN features
        X = np.zeros(shape=(len(features), len(features)), dtype=float)  # distance matrix
        pairs = list(combinations_with_replacement(features, 2))        # all possible combinations

        if self._nmi_cache:
            # ignore unselected features in cache
            cache = [(pair, nmi) for pair, nmi in self._nmi_cache if pair[0] in features and pair[1] in features]
            # add each cached nmi to the distance matrix
            for cached_pair, nmi in cache:
                # remove cached_pair from pairs
                pairs = filter(lambda pair: (pair[0] == cached_pair[0] and pair[1] == cached_pair[1]) or
                                            (pair[0] == cached_pair[1] and pair[1] == cached_pair[0]), pairs)
                # add cached nmi to matrix
                i, j = features.index(cached_pair[0]), features.index(cached_pair[1])
                X[i][j] = 1 - nmi
                X[j][i] = 1 - nmi

        # map pairs to nmi
        if self.pool is None:
            results = map(self._estimate_nmi, pairs)
        else:
            results = self.pool.imap(self._estimate_nmi, pairs)
            self.pool.close()

        # fill matrix with pair nmi values
        count = 0
        for pair, nmi in zip(pairs, results):

            # print progress updates
            count += 1
            if len(pairs) % (len(pairs)*0.05) == 0:
                logger.info("Progress: {}/{}".format(count, len(pairs)))

            fidx1, fidx2 = pair
            i, j = features.index(fidx1), features.index(fidx2)
            X[i][j] = 1 - nmi
            X[j][i] = 1 - nmi

        # restart pool if multiprocessing
        if self.pool is not None:
            self.pool.join()
            self.pool.restart()

        # use DBSCAN to cluster our data
        labels = DBSCAN(eps=eps, metric='precomputed').fit_predict(X)
        logger.info("Found {} clusters.".format(set(labels)))

        # organize the topN features into sub-lists where
        # each sub-list contains all features in a cluster
        clusters = []
        for label in range(min(labels), max(labels)):
            cluster = [features[i] for i, la in enumerate(labels) if la == label]
            clusters.append(cluster)

        logger.debug("Clusters: {}".format(labels))
        return clusters
