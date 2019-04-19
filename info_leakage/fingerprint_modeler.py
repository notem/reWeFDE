
import math
from data_utils import logger
from collections.abc import Iterable


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
        for site, mkde in zip(self.data.sites, mkdes):

            # n = k * pr(c[i]) -- number of samples per site
            num = int(sample_size*web_priors[site])

            # sample from pdf(f|c[i])
            x = mkde.sample(n_samples=num)
            samples.extend(x)

        return samples

    def _estimate_entropy(self, sample, mkdes, website_priors):
        """

        :return:
        """
        # Shannon Entropy func: -p(x)*log2(p(x))
        h = lambda x: -x * math.log2(x)

        # get probabilities of sample from each website density distribution
        marginal_probs = [mkde.predict([sample]) for mkde in mkdes]

        # take the log2 of probs (copy original WeFDE code)
        marginal_probs = [math.log2(prob) if prob > 0 else -300.0 for prob in marginal_probs]

        # don't know why this is done
        marginal_probs = [prob - max(marginal_probs) for prob in marginal_probs]

        # reverse log2
        marginal_probs = [2**prob for prob in marginal_probs]

        # weight by website priors
        prob_temp = [prob*prior for prob, prior in zip(marginal_probs, website_priors)]

        # normalize probabilities?
        prob_indiv = [prob / sum(prob_temp) for prob in prob_temp]

        # pointless check, previous line guarantees this to be true
        assert(sum(prob_indiv) > 0.99)

        # sum the entropy of site probabilities
        entropy = sum([h(prob) for prob in prob_indiv if prob > 0])
        return entropy

    def information_leakage(self, features):
        """
        Evaluate the information leakage.
        Computes marginal KDEs for features given a sites using the awkde library.
        Conditional entropy estimated via monte-carlo integration.
        :param features: index of feature(s) whose joint leakage should be produced
        """
        # catch unhandled errors
        try:
            # convert one feature to singular list for comparability
            if not isinstance(features, Iterable):
                features = [features]

            logger.debug("Measuring leakage for {}".format(features))

            # create pdf for sampling and probability calculations
            mkdes = [self.data.model_distribution(features=features, site=site) for site in self.data.sites]

            # Shannon Entropy func: -p(x)*log2(p(x))
            h = lambda x: -x * math.log2(x)

            # H(C) -- compute website entropy
            website_priors = [1/len(self.data.sites) for _ in self.data.sites]
            sequence = [h(prior) for prior in website_priors if prior > 0]
            H_C = sum(sequence)

            # performing sampling for monte-carlo evaluation of H(C|f)
            samples = self._sample(mkdes, website_priors, self.sample_size)

            # ----------------------------------------------------------------
            # BEGIN SECTION: Following algorithm copied from original WeFDE code
            # ----------------------------------------------------------------
            # compute the log of the probabilities for each sample
            prob_set = []
            for sample in samples:

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

        except Exception:
            # in cases where there is an unknown error, save leakage as N/A
            # ignore these features when computing combined leakage
            logger.exception("Exception when estimating leakage for {}.".format(features))
            return None
