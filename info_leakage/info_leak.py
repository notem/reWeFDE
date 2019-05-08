"""
Main project file which performs info-leak measure
"""
import argparse
import sys
import pickle
import os
import signal
from pathos.multiprocessing import ProcessPool as Pool

from fingerprint_modeler import WebsiteFingerprintModeler
from mi_analyzer import MutualInformationAnalyzer
from data_utils import load_data, WebsiteData, logger


def individual_measure(fingerprinter, pool=None):
    """
    :param fingerprinter:
    :param pool:
    :return:
    """
    # if a pool has been provided, perform computation in parallel
    # otherwise do serial computation
    if pool is None:
        proc_results = map(fingerprinter, fingerprinter.data.features)
    else:
        proc_results = pool.imap(fingerprinter, fingerprinter.data.features)
        pool.close()
    size = len(fingerprinter.data.features)  # number of features

    logger.info("Begin individual leakage measurements.")
    # measure information leakage
    # log current progress at twenty intervals
    leakage_indiv = []
    for leakage in proc_results:
        if len(leakage_indiv) % int(size*0.05) == 0:
            logger.info("Progress: {}/{}".format(len(leakage_indiv), size))
        leakage_indiv.append(leakage)
    if pool is not None:
        pool.join()
        pool.restart()
    return leakage_indiv


def combined_measure(analyzer, fingerprinter, pool=None):
    """
    :param analyzer:
    :param fingerprinter:
    :param pool:
    :return:
    """
    # process into list of non-redundant features
    logger.info("Begin feature pruning.")
    pruned = analyzer.prune()

    # cluster non-redundant features
    logger.info("Begin feature clustering.")
    clusters = analyzer.cluster(pruned)

    # if a pool has been provided, perform computation in parallel
    # otherwise do serial computation
    if pool is None:
        proc_results = map(fingerprinter, clusters)
    else:
        proc_results = pool.imap(fingerprinter, clusters)
        pool.close()

    logger.info("Begin cluster leakage measurements.")
    # measure information for each cluster
    # log current progress at twenty intervals
    leakage_joint = []
    for leakage in proc_results:
        if len(leakage_joint) % int(len(clusters)*0.05) == 0:
            logger.info("Progress: {}/{}".format(len(leakage_joint), len(clusters)))
        leakage_joint.append(leakage)
    if pool is not None:
        pool.join()
        pool.restart()
    return leakage_joint


def parse_args():
    """
    Parse command line arguments
    Accepted arguments:
      (f)eatures   -- directory which contains feature files
      (i)ndividual -- pickle file where individual leakage is (to be) saved
      (c)ombined   -- pickle file where the combined leakage is to be saved
    """
    parser = argparse.ArgumentParser("TODO: Program description")

    # Required Arguments
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

    # Optional Arguments
    # number of samples for monte-carlo integration
    parser.add_argument("--n_samples",
                        type=int,
                        default=5000,
                        help="The number of samples to use when performing Monte-Carlo Integration estimation. "
                             "Higher values result in more accurate measurements, but longer runtimes.")
    # redundancy threshold
    parser.add_argument("--nmi_threshold",
                        type=int,
                        default=0.9,
                        help="The theshold value used to identify redundant features. "
                             "A value between 0.0 and 1.0.")
    parser.add_argument("--topn",
                        type=int,
                        default=100,
                        help="The number of top features to save during combined feature analysis")
    # number of processes
    parser.add_argument("--n_procs",
                        type=int,
                        default=1,
                        help="The number of processes to use when performing parallel operations")
    return parser.parse_args()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(args):
    """
    execute main logic
    """
    # prepare feature dataset
    logger.info("Loading dataset.")
    X, Y = load_data(args.features)
    feature_data = WebsiteData(X, Y)
    logger.info("Loaded {} instances.".format(len(feature_data)))

    # create process pool
    pool = Pool(args.n_procs, initializer=init_worker)

    # initialize fingerprint modeler
    fingerprinter = WebsiteFingerprintModeler(feature_data,
                                              sample_size=args.n_samples)

    # perform individual information leakage measurements
    leakage_indiv = None
    if args.individual:

        # load previous leakage measurements if possible
        if os.path.exists(args.individual):
            with open(args.individual, "rb") as fi:
                logger.info("Loading saved individual leakage measures.")
                leakage_indiv = pickle.load(fi)

        # otherwise do individual measure
        else:
            leakage_indiv = individual_measure(fingerprinter, pool)

            # save individual leakage to file
            logger.info("Saving individual leakage to {}.".format(args.individual))
            if os.path.dirname(args.individual):
                os.makedirs(os.path.dirname(args.individual))
            with open(args.individual, "wb") as fi:
                pickle.dump(leakage_indiv, fi)

    # perform combined information leakage measurements
    leakage_joint = None
    if args.combined:

        # load joint leakage file
        if os.path.exists(args.combined):
            with open(args.combined, "rb") as fi:
                logger.info("Loading saved joint leakage measures.")
                leakage_joint = pickle.load(fi)

        # otherwise do joint leakage estimation
        else:
            # initialize MI analyzer
            analyzer = MutualInformationAnalyzer(feature_data,
                                                 leakage_indiv,
                                                 nmi_threshold=args.nmi_threshold,
                                                 topn=args.topn,
                                                 pool=pool)

            # perform combined information leakage analysis
            leakage_joint = combined_measure(analyzer, fingerprinter, pool)

            # save individual leakage to file
            logger.info("Saving joint leakage to {}.".format(args.combined))
            if os.path.dirname(args.combined):
                os.makedirs(os.path.dirname(args.combined))
            with open(args.combined, "wb") as fi:
                pickle.dump(leakage_joint, fi)

    # summarize results
    # TODO

    logger.info("Finished execution.")


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(-1)


