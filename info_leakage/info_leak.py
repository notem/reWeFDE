"""
Main project file which performs info-leak measure
"""
import argparse
import sys
import pickle

from data_utils import load_data
from fingerprint_modeler import WebsiteFingerprintModeler

from data_utils import logger


def info_leakage(X, Y):
    """
    Compute individual information leakage of features
    """
    fingerprinter = WebsiteFingerprintModeler(X, Y)

    # evaluate leakage for each feature
    leakage = []
    for i in range(X.shape[1]):
        logger.info("Evaluating leakage for feature #{}.".format(i+1))
        try:
            leakage.append(fingerprinter.individual_leakage(i))
        except KeyboardInterrupt:
            sys.exit(-1)
        except Exception:
            logger.exception("Exception when estimating leakage for feature #{}.".format(i+1))
            leakage.append(None)

    # index of leakage maps to the feature number
    return leakage


def parse_args():
    """
    Parse command line arguments
    Accepted arguments:
      (t)races -- directory which contains feature files
    """
    parser = argparse.ArgumentParser("")
    parser.add_argument("-t", "--traces")
    return parser.parse_args()


def main(args):
    """
    execute main logic
    """

    logger.info("Loading dataset.")
    X, Y = load_data(args.traces)

    logger.info("Begin info-leakage evaluation.")
    leakage = info_leakage(X, Y)

    logger.info("Saving leakage as pickle file.")
    logger.debug(leakage)
    pickle.dump(leakage, open("leakage.pkl", "wb"))


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(-1)


