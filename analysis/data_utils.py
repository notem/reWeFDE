# -*- coding: utf-8 -*-
import csv
import os
import sys
import numpy as np
import logging as log
import dill


# set to True to enable debugging information
DEBUG_ON = False


def ready_logger():
    """
    Prepare python logging object.

    Logger is configured to print DEBUG & INFO messages to stdout.
    ERROR and WARNING logs are printed to stderr.
    Modify this function to customize logging behavior

    Returns
    -------
    Logger
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
    logging_handler_out.setLevel(log.DEBUG if DEBUG_ON else log.INFO)
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


logger = ready_logger()
"""Logger: logging object to be used throughout the application

The logger object is built using the ready_logger() function, 
and is used as a mechanism to control output to standard output/error.
The design of this is that the logger object should be imported and used by all other scripts in the module.
"""


class WebsiteData(object):
    """
    Object-wrapper to conveniently manage dataset
    """
    def __init__(self, X, Y):
        self._X = X
        self._Y = Y
        self.features = list(range(X.shape[1]))
        self.sites = list(range(len(np.unique(self._Y))))

    def __len__(self):
        return self._X.shape[0]

    def get_labels(self):
        """
        Return Y

        Returns
        -------
        ndarray

        """
        return self._Y

    def get_site(self, label, feature=None):
        """
        Return X for given site.

        Parameters
        ----------
        label : int
            The site label to load
        feature : int
            The feature number to load.
            Load all features if None.

        Returns
        -------
        ndarray

        """
        f = [True if y == label else False for y in self._Y]
        if feature is not None:
            return self._X[f, feature]
        return self._X[f, :]

    def get_feature(self, feature, site=None):
        """
        Return all X for a specific feature

        Parameters
        ----------
        feature : int
            The feature which to load.
        site : int
            The site which to load.
            Load from all sites if None.

        Returns
        -------
        ndarray

        """
        if site is not None:
            f = [True if y == site else False for y in self._Y]
            return self._X[f, feature]
        return self._X[:, feature]


def load_data(directory, extension='.features', delimiter=' ', split_at='-',
              max_classes=99999, max_instances=99999, pack_dataset=True):
    """
    Load feature files from a directory.

    Parameters
    ----------
    directory : str
        System file path to a directory containing feature files.
    extension : str
        File extension used to identify feature files.
    delimiter : str
        Character string used to split features in the feature files.
    split_at : str
        Character string used to split feature file names.
        First substring identifies the class, while the second substring identifies the instance number.
        Instance number is ignored.
    max_classes : int
        Maximum number of classes to load.
    max_instances : int
        Maximum number of instances to load per class.
    pack_dataset : bool
        Determines whether or not ascii feature files should be packed into a condensed pickle file.
        If True, the function will attempt to load from the packed feature file as well.
        The packed feature file is saved in the root of the same directory as the feature files.

    Returns
    -------
    ndarray
        Numpy array of Nxf containing site visit feature instances.
    ndarray
        Numpy array of Nx1 containing the labels for site visits.

    """
    # load pickle file if it exist
    feat_pkl = os.path.join(directory, "features.pkl")
    if os.path.exists(feat_pkl) and pack_dataset:
        with open(feat_pkl, "rb") as fi:
            X, Y = dill.load(fi)
            return X, Y
    else:
        X = []  # feature instances
        Y = []  # site labels
        for root, dirs, files in os.walk(directory):

            # filter for feature files
            files = [fi for fi in files if fi.endswith(extension)]

            def isfloat(element):
                """
                Simple function to reliably determine if a string element is a float.
                Used for feature file filtering.
                """
                try:
                    float(element)
                    return True
                except ValueError:
                    return False

            # read each feature file as CSV
            class_counter = dict()  # track number of instances per class
            for file in files:

                # feature files are of name
                cls, ins = file.split(split_at)

                # skip if maximum number of instances reached
                if class_counter.get(cls, 0) >= max_instances:
                    continue

                # skip if maximum number of classes reached
                if int(cls) >= max_classes:
                    continue

                with open(os.path.join(root, file), "r") as csvFile:

                    # load the csv file and parse it into a data instance
                    features = list(csv.reader(csvFile, delimiter=delimiter))
                    features = [[float(f) if isfloat(f) else 0 for f in instance if f] for instance in features]

                    # cut off instance count is above the maximum
                    features = features[:max_instances - class_counter.get(int(cls), 0)]

                    X.extend(features)
                    Y.extend([int(cls)-1 for _ in range(len(features))])
                    class_counter[int(cls)] = class_counter.get(int(cls), 0) + len(features)

        # adjust labels such that they are assigned a number from 0..N
        # (required when labels are non-numerical or does not start at 0)
        # try to keep the class numbers the same if numerical
        labels = list(set(Y))
        labels.sort()
        d = dict()
        for i in range(len(labels)):
            d[labels[i]] = i
        Y = list(map(lambda x: d[x], Y))

        # save dataset to pickle file for quicker future loading
        if pack_dataset:
            X, Y = np.array(X), np.array(Y)
            with open(feat_pkl, "wb") as fi:
                dill.dump((X, Y), fi)

    # return X and Y as numpy arrays
    return X, Y

