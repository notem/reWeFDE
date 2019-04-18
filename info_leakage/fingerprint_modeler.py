
import math
import numpy as np
from awkde.awkde import GaussianKDE

from data_utils import WebsiteData
from data_utils import logger


class WebsiteFingerprintModeler(object):

    def __init__(self, X, Y):
        """
        :param X: instances as numpy matrix
        :param Y: labels
        """
        self.data = WebsiteData(X, Y)

    def _model_individual(self, feature, site=None):
        """
        Produce AKDE for a single feature or single feature for a particular site.
        """
        if site:
            # pdf(f|c)
            X = self.data.get_site(site, feature)
        else:
            # pdf(f)
            X = self.data.get_feature(feature)
        X = np.reshape(X, (X.shape[0], 1))

        kde = GaussianKDE()
        try:
            kde.fit(X)

        # AWKDE cannot model data whose distribution is a single value
        # This results in a linear algebra error during fit()
        # To remedy this, add a negligible value to the first feature instance
        except np.linalg.LinAlgError:
            X[0][0] += 0.000001
            kde.fit(X)

        return kde

    def _sample(self, mkdes, web_priors, sample_size):
        """
        Select samples for monte-carlo evaluation.
        """
        samples = []
        for site, mkde in zip(self.data.sites, mkdes):

            # n = k * pr(c[i]) -- number of samples per site
            num = int(sample_size*web_priors[site])

            # sample from pdf(f|c[i])
            x = mkde.sample(n_samples=num)
            samples.extend(x)

        return samples

    def individual_leakage(self, feature, sample_size=5000):
        """
        Evaluate the information leakage.
        Computes joint KDE of feature and sites using the awkde library.
        Performance linear monte-carlo integration to estimate the
          conditional entropy of the sites given the feature.
        """
        # create pdf for sampling and probability calculations
        mkdes = [self._model_individual(feature=feature, site=site) for site in self.data.sites]

        # Shannon Entropy func: -p(x)*log2(p(x))
        h = lambda x: -x * math.log2(x)

        # H(C) -- compute website entropy
        website_priors = [1/len(self.data.sites) for _ in self.data.sites]
        sequence = [h(prior) for prior in website_priors if prior > 0]
        H_C = sum(sequence)

        # performing sampling for monte-carlo evaluation of H(C|f)
        samples = self._sample(mkdes, website_priors, sample_size)

        # ----------------------------------------------------------------
        # BEGIN SECTION: Following algorithm copied from original WeFDE code
        # ----------------------------------------------------------------
        # compute the log of the probabilities for each sample
        prob_set = []
        for i, sample in enumerate(samples):

            # get probabilities of sample from each website density distribution
            marginal_probs = [mkde.predict([sample]) for mkde in mkdes]

            # take the log2 of probs (copy original WeFDE code)
            marginal_probs = [math.log2(prob) if prob > 0 else -300.0 for prob in marginal_probs]
            prob_set.append(marginal_probs)

        # don't know what this does
        prob_set = [[prob - max(prob_inst) for prob in prob_inst]
                    for prob_inst in prob_set]

        # reverse log2
        prob_set = [[2**prob for prob in prob_inst]
                    for prob_inst in prob_set]

        # weight by website priors
        prob_temp = [[prob*prior for prob, prior in zip(prob_inst, website_priors)]
                     for prob_inst in prob_set]

        # normalize probabilities?
        prob_indiv = [[prob / sum(prob_inst) for prob in prob_inst]
                      for prob_inst in prob_temp]

        # pointless check, previous line guarantees this to be true
        for prob_inst in prob_indiv:
            assert(sum(prob_inst) > 0.99)

        # ----------------------------------------------------------------
        # END SECTION
        # ----------------------------------------------------------------

        # compute entropy for instances
        entropies = [sum([h(prob) for prob in prob_inst if prob > 0])
                     for prob_inst in prob_indiv]

        # H(C|f) -- compute conditional entropy via monte-carlo from sample probabilities
        # see: https://en.wikipedia.org/wiki/Monte_Carlo_integration#Importance_sampling_algorithm
        H_CF = sum(entropies)/len(entropies)

        # I(C;f) = H(C) - H(C|f) -- compute information leakage
        # see: https://en.wikipedia.org/wiki/Mutual_information
        leakage = H_C - H_CF

        # debug output
        logger.debug("{l} = {c} - {cf}"
                     .format(l=leakage, c=H_C, cf=H_CF))
        return leakage
