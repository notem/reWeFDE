import math
import numpy as np
from data_utils import logger


class MutualInformationAnalyzer(object):

    def __init__(self, data, nmi_threshold=0.9, topn=100, n_samples=1000):
        self.data = data
        self.nmi_threshold = nmi_threshold
        self.topn = topn
        self.n_samples = n_samples

    def _calculate_nmi(self, c, r):
        """
        :param c:
        :param r:
        :return:
        """
        # model the joint and marginal PDFs of features
        jkde = self.data.model_distribution([c, r])
        mkde_c = self.data.model_distribution(c)
        mkde_r = self.data.model_distribution(r)

        # determine the domains of each feature
        domain_c = (np.amin(self.data.get_feature(c)), np.amax(self.data.get_feature(c)))
        domain_r = (np.amin(self.data.get_feature(r)), np.amax(self.data.get_feature(r)))

        # entropy function
        h = lambda x: -x * math.log2(x) if x > 0 else 0

        # perform uniform monte-carlo integration estimation:
        # TODO: replace with a more efficient estimation method
        # - perform uniform sampling over the domains of each feature
        samples = zip(np.random.uniform(*domain_c, self.n_samples).tolist(),
                      np.random.uniform(*domain_r, self.n_samples).tolist())

        # - calculate entropy of each sample
        H_c = [h(mkde_c.predict([sample])) for sample in samples]
        H_r = [h(mkde_r.predict([sample])) for sample in samples]
        H_cr = [h(jkde.predict([sample])) for sample in samples]

        # - sum and divide
        H_c = sum(H_c) / len(H_c)
        H_r = sum(H_r) / len(H_r)
        H_cr = sum(H_cr) / len(H_cr)

        # I(c;r) = H(c) + H(r) - H(c,r)
        I_cr = H_c + H_r - H_cr

        # NMI = I(c;r) / max(H(c),H(r))
        nmi = I_cr / max(H_c, H_r)

        # return normalized pair-wise mutual information
        return nmi

    def prune(self, leakage_indiv):
        """

        :param leakage_indiv:
        :return:
        """
        # list of best features
        pruned = []

        # sort the list of features by their individual leakage
        # we will process these features in the order of their importance
        tuples = zip(range(len(leakage_indiv)), leakage_indiv)
        tuples = sorted(tuples, key=lambda x: (-x[1], x[0]))

        # continue to process features until either there are no features left to process
        # or the topN features have been selected
        while tuples and len(pruned) < self.topn:

            # the next most important feature
            current_feature = tuples.pop()[0]
            logger.debug("MI analysis on feature #{}".format(current_feature))

            # for all top features, measure pair-wise mutual information to check for redundancy
            is_redundant = False
            for count, tup in enumerate(pruned):
                feature = tup[0]

                # calculate the normalized mutual information
                nmi = self._calculate_nmi(current_feature, feature)

                # prune if nmi is above threshold
                if nmi > self.nmi_threshold:
                    is_redundant = True
                    logger.debug("Feature #{} is redundant w/ #{}".format(current_feature, feature))
                    break

            # if the current feature does not appear to be redundant with any other top features,
            # add current feature to top features list
            if not is_redundant:
                pruned.append(current_feature)

        return pruned

    def cluster(self, features):
        """

        :param features:
        :return:
        """
        # TODO: cluster features using DBSCAN
        return [[feature] for feature in features]
