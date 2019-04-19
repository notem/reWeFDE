import csv
import os
import sys

import numpy as np
import logging as log
from awkde.awkde import GaussianKDE
from collections.abc import Iterable


def ready_logger():
    """
    Prepare python logging object.
    Modify this function to customize logging behavior
    """
    class LessThanFilter(log.Filter):
        def __init__(self, exclusive_maximum, name=""):
            super(LessThanFilter, self).__init__(name)
            self.max_level = exclusive_maximum

        def filter(self, record):
            # non-zero return means we log this message
            return 1 if record.levelno < self.max_level else 0

    # Get the root logger
    logger = log.getLogger()

    # Have to set the root logger level, it defaults to logging.WARNING
    logger.setLevel(log.NOTSET)

    logging_handler_out = log.StreamHandler(sys.stdout)
    logging_handler_out.setLevel(log.DEBUG)
    logging_handler_out.addFilter(LessThanFilter(log.WARNING))

    logging_handler_err = log.StreamHandler(sys.stderr)
    logging_handler_err.setLevel(log.WARNING)

    format = log.Formatter('[%(asctime)s][%(processName)-10s][%(levelname)s] %(message)s')
    logging_handler_err.setFormatter(format)
    logging_handler_out.setFormatter(format)

    # add handlers to logger
    logger.addHandler(logging_handler_err)
    logger.addHandler(logging_handler_out)

    # return the prepared logger
    return logger


# logging object to be used throughout the application
logger = ready_logger()


class WebsiteData(object):
    """
    Object-wrapper to conveniently manage dataset
    """
    def __init__(self, X, Y):
        self._X = X
        self._Y = Y
        self.features = range(X.shape[1])
        self.sites = range(len(np.unique(self._Y)))

    def get_labels(self):
        """
        Return Y
        """
        return np.copy(self._Y)

    def get_site(self, label, feature=None):
        """
        Return X for given site. Optionally, also filter by feature.
        """
        f = [True if y == label else False for y in self._Y]
        if feature is not None:
            return np.copy(self._X[f, feature])
        return np.copy(self._X[f, :])

    def get_feature(self, feature):
        """
        Return all X for a specific feature
        """
        return np.copy(self._X[:, feature])

    def model_distribution(self, features, site=None):
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
            if site:    # pdf(f|c)
                X_f = self.get_site(site, feature)
            else:       # pdf(f)
                X_f = self.get_feature(feature)
            if X:
                np.hstack(X, X_f)
            else:
                X = np.reshape(X_f, (X_f.shape[0], 1))

        # fit KDE on X
        kde = GaussianKDE()
        try:
            kde.fit(X)

        # AWKDE cannot model data whose distribution is a single value
        # This results in a linear algebra error during fit()
        # To remedy this, add negligible value of features in the first instance
        except np.linalg.LinAlgError:
            for d in range(X.shape[1]):
                X[0][d] += 0.00000001
            kde.fit(X)

        return kde


def load_data(directory, extension='.features', delimiter=' '):
    """
    Load feature files from feature directory
    :return X - numpy array of data instances w/ shape (n,f)
    :return Y - numpy array of data labels w/ shape (n,1)
    """
    X = []  # feature instances
    Y = []  # site labels
    for root, dirs, files in os.walk(directory):

        # filter for feature files
        files = [fi for fi in files if fi.endswith(extension)]

        # read each feature file as CSV
        for file in files:
            cls, ins = file.split("-")
            with open(os.path.join(root, file), "r") as csvFile:
                features = [float(f) for f in list(csv.reader(csvFile, delimiter=delimiter))[0] if f]
                X.append(features)
                Y.append(int(cls))

    # return X and Y as numpy arrays
    return np.array(X), np.array(Y)

