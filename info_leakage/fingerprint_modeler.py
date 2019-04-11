
import math
import numpy as np
from awkde.awkde import GaussianKDE
from skmonaco import mcimport, mcquad, mcmiser


class WebsiteData(object):
    """
    Object-wrapper to conveniently manage dataset
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.features = range(X.shape[1])
        self.sites = range(len(np.unique(self.Y)))

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
        Produce AKDE for a single feature or single feature for a particular site
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

    def _model_joint(self, feature):
        """
        Produce AKDE for joint distributions
        """
        X = self.data.X[:, feature]
        X = np.reshape(X, (X.shape[0], 1))
        Y = self.data.Y
        Y = np.reshape(Y, (Y.shape[0], 1))

        joint = np.hstack((Y, X))

        kde = GaussianKDE()
        kde.fit(joint)
        return kde

    def _sample(self, feature, web_priors, sample_size):
        """
        Select samples for monte-carlo evaluation.
        """
        samples = []
        for i, site in enumerate(self.data.sites):

            # n = k * pr(c[i]) -- number of samples per site
            num = int(sample_size*web_priors[i])

            try:
                # distribution pdf(f|c[i])
                kde = self._model_individual(feature=feature, site=site)

                # sample from pdf(f|c[i])
                x = kde.sample(n_samples=num)
                samples.extend(x)

            except np.linalg.LinAlgError:
                pass    # don't sample if KDE creation failed

        return samples

    def individual_leakage(self, feature, sample_size=5000):
        """
        Evaluate the information leakage.
        Computes joint KDE of feature and sites using the awkde library.
        Performance linear monte-carlo integration to estimate the
          conditional entropy of the sites given the feature.
        """
        # create pdf for sampling and probability calculations
        jkde = self._model_joint(feature)

        # Shannon Entropy func: -p(x)*log2(p(x))
        h = lambda x: -x * math.log2(x)

        # H(C) -- compute website entropy
        website_priors = [1/len(self.data.sites) for _ in self.data.sites]
        sequence = [h(prior) for prior in website_priors if prior > 0]
        H_C = sum(sequence)

        # performing sampling for monte-carlo evaluation of H(C|f)
        samples = self._sample(feature, website_priors, sample_size)    # author's method
        #samples = mkde.sample(sample_size)                              # from generic distribution
        #samples = [l[1] for l in jkde.sample(sample_size)]

        # compute conditional entropy for each sample
        entropies = []
        for i, sample in enumerate(samples):
            print("=> generating monte-carlo sample: {}".format(i), end="\r")

            # p(x,y) -- compute joint probability for all sites
            joint_probs = [jkde.predict([site, sample]).tolist() for site in self.data.sites]
            joint_probs = [prob for sublist in joint_probs for prob in sublist]

            # p(x) -- compute marginal probability
            marginal_prob = sum([jkde.predict([site, sample]) for site in self.data.sites])

            # p(y|x) = p(x,y)/p(x) -- compute conditional probability
            # see: https://en.wikipedia.org/wiki/Conditional_probability_distribution
            conditional_probs = [prob/marginal_prob for prob in joint_probs]

            # if conditional pdf does not sum to 1.0, something is wrong!
            assert(sum(conditional_probs) >= 0.99)

            # compute entropy
            # see: https://en.wikipedia.org/wiki/Conditional_entropy
            entropy = sum([h(cprob) for cprob in conditional_probs])
            entropies.append(entropy)

        # H(C|f) -- compute conditional entropy via monte-carlo
        # see: https://en.wikipedia.org/wiki/Monte_Carlo_integration#Importance_sampling_algorithm
        H_CF = sum(entropies)/len(entropies)

        # I(C;f) = H(C) - H(C|f) -- compute information leakage
        # see: https://en.wikipedia.org/wiki/Mutual_information
        leakage = H_C - H_CF

        # debug output
        print("=> H(C) = {}             ".format(H_C))
        print("=> H(C|f) = {}".format(H_CF))
        print("=> I(C;f) = {}".format(leakage))
        return leakage

    def individual_leakage_multi(self, feature, sample_size=1000, proc_num=8):
        """
        Evaluate the information leakage.
        Improve computation performance by utilizing multiple processes during monte-carlo integration.
        Monte-carlo integration is performed using the skmonaco library.
        """
        # create pdf for sampling and probability calculations
        jkde = self._model_joint(feature)

        # Shannon Entropy func: -p(x)*log2(p(x))
        h = lambda x: -x * math.log2(x)

        # H(C) -- compute website entropy
        website_priors = [1/len(self.data.sites) for _ in self.data.sites]
        sequence = [h(prior) for prior in website_priors if prior > 0]
        H_C = sum(sequence)

        def f(x):
            """
            Function to integrate by Monte-Carlo using skmonaco library.
            """
            if np.ndim(x) == 1:
                samples = [x]
            else:
                samples = []
                for sample_no in range(len(x)):
                    samples.append(x[sample_no, :])

            entropies = []
            for sample in samples:
                # p(x,y) -- compute joint probability for all sites
                joint_probs = [jkde.predict([site, sample]).tolist() for site in self.data.sites]
                joint_probs = [prob for sublist in joint_probs for prob in sublist]

                # p(x) -- compute marginal probability
                marginal_prob = sum([jkde.predict([site, sample]) for site in self.data.sites])

                # p(y|x) = p(x,y)/p(x) -- compute conditional probability
                # see: https://en.wikipedia.org/wiki/Conditional_probability_distribution
                conditional_probs = [prob/marginal_prob for prob in joint_probs]

                # if conditional pdf does not sum to 1.0, something is wrong!
                assert(sum(conditional_probs) >= 0.99)

                # compute entropy
                # see: https://en.wikipedia.org/wiki/Conditional_entropy
                entropy = sum([h(cprob) for cprob in conditional_probs])
                entropies.append(entropy)

            if len(entropies) == 1:
                return entropies[0]
            return entropies

        def d(size):
            """
            Distribution to sample from using skmonaco library.
            """
            if isinstance(size, tuple):
                samples = jkde.sample(size[0])
                return samples[:, 1]
            else:
                samples = jkde.sample(size)
                return samples[:, 1]

        # H(C|f) -- compute conditional entropy via monte-carlo
        # see: https://en.wikipedia.org/wiki/Monte_Carlo_integration#Importance_sampling_algorithm
        H_CF = mcimport(f, sample_size, d, nprocs=proc_num, batch_size=(sample_size//proc_num))[0]

        # I(C;f) = H(C) - H(C|f) -- compute information leakage
        # see: https://en.wikipedia.org/wiki/Mutual_information
        leakage = H_C - H_CF

        # debug output
        print("=> H(C) = {}             ".format(H_C))
        print("=> H(C|f) = {}".format(H_CF))
        print("=> I(C;f) = {}".format(leakage))
        return leakage
