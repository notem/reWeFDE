"""
Main project file which performs info-leak measure
"""
import argparse
import sys
import pickle
import os

from multiprocessing import Pool
from data_utils import load_data, WebsiteData
from fingerprint_modeler import WebsiteFingerprintModeler
from mi_analyzer import MutualInformationAnalyzer
from data_utils import logger


def parse_args():
    """
    Parse command line arguments
    Accepted arguments:
      (f)eatures   -- directory which contains feature files
      (i)ndividual -- pickle file where individual leakage is (to be) saved
      (c)ombined   -- pickle file where the combined leakage is to be saved
    """
    parser = argparse.ArgumentParser("TODO: Program description")

    # directory containing feature files
    parser.add_argument("-f", "--features",
                        required=True,
                        type=str,
                        help="Directory which contains files with the .feature extension.")

    # location to save individual measurements,
    # or location from which to load measurements
    parser.add_argument("-i", "--individual",
                        type=str,
                        help="The file used to save or load individual leakage measurements.")

    # location to save combined measurements,
    # or location from which to load measurements
    parser.add_argument("-c", "--combined",
                        type=str,
                        help="The file used to save or load combined leakage measurements.")

    # number of samples for monte-carlo integration
    parser.add_argument("-n", "--n_samples",
                        type=int,
                        default=5000,
                        help="The number of samples to use when performing Monte-Carlo Integration estimation. "
                             "Higher values result in more accurate measurements, but longer runtimes.")

    # number of processes to spawn when multiprocessing
    parser.add_argument("-p", "--n_procs",
                        type=int,
                        default=os.cpu_count(),
                        help="The number of processors to use when performing concurrent functions.")
    return parser.parse_args()


def main(args):
    """
    execute main logic
    """
    # prepare feature dataset
    logger.info("Loading dataset.")
    X, Y = load_data(args.features)
    data = WebsiteData(X, Y)

    # initialize fingerprint modeler
    fingerprinter = WebsiteFingerprintModeler(data, sample_size=args.n_samples)

    # perform individual information leakage measurements
    leakage_indiv = []
    if args.individual:

        # load previous individual leakage measurements if possible
        if os.path.exists(args.individual):
            with open(args.individual, "rb") as fi:
                logger.info("Loading saved individual leakage measures.")
                leakage_indiv = pickle.load(fi)

        else:
            # for every feature, calculate leakage
            logger.info("Begin individual leakage measurements.")

            # measure information for each cluster
            # log current progress at twenty intervals
            with Pool(args.n_procs) as pool:
                logger.info("Progress: {}/{}".format(0, X.shape[1]))
                iter = pool.imap(fingerprinter.information_leakage, range(X.shape[1]))
                for i, leakage in enumerate(iter):
                    if (i+1) % int(X.shape[1]*0.05) == 0:
                        logger.info("Progress: {}/{}".format(i+1, X.shape[1]))
                    leakage_indiv.append(leakage)

            # save individual leakage to file
            logger.info("Saving individual leakage to {}.".format(args.individual))
            if os.path.dirname(args.individual):
                os.makedirs(os.path.dirname(args.individual))
            with open(args.individual, "wb") as fi:
                pickle.dump(leakage_indiv, fi)

    # perform combined information leakage measurements
    leakage_joint = []
    if args.combined:

        if os.path.exists(args.combined):
            with open(args.combined, "rb") as fi:
                logger.info("Loading saved joint leakage measures.")
                leakage_joint = pickle.load(fi)
        else:
            # initialize MI analyzer
            analyzer = MutualInformationAnalyzer(data, nmi_threshold=0.9, topn=100)

            # process into list of non-redundant features
            logger.info("Begin feature pruning.")
            pruned = analyzer.prune(leakage_indiv)

            # cluster non-redundant features
            logger.info("Begin feature clustering.")
            clusters = analyzer.cluster(pruned)

            # measure information for each cluster
            # log current progress at twenty intervals
            with Pool(args.n_procs) as pool:
                logger.info("Progress: {}/{}".format(0, len(clusters)))
                iter = pool.imap(fingerprinter.information_leakage, clusters)
                for i, leakage in enumerate(iter):
                    if (i+1) % int(len(clusters)*0.05) == 0:
                        logger.info("Progress: {}/{}".format(i+1, len(clusters)))
                    leakage_joint.append(leakage)

            # save individual leakage to file
            logger.info("Saving joint leakage to {}.".format(args.combined))
            if os.path.dirname(args.combined):
                os.makedirs(os.path.dirname(args.combined))
            with open(args.combined, "wb") as fi:
                pickle.dump(leakage_joint, fi)

    # print summary
    # TODO

    logger.info("Finished execution.")


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(-1)


