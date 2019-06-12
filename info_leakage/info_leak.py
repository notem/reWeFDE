# -*- coding: utf-8 -*-
"""
Main project file which performs info-leak measure
"""
import argparse
import sys
import dill
import os
from pathos.multiprocessing import cpu_count
from pathos.multiprocessing import ProcessPool as Pool
from fingerprint_modeler import WebsiteFingerprintModeler
from mi_analyzer import MutualInformationAnalyzer
from data_utils import load_data, WebsiteData, logger


def individual_measure(fingerprinter, pool=None, checkpoint=None):
    """
    Perform information leakage analysis for each feature one-by-one.

    The resulting leakages can be saved in a plain-text ascii checkpoint file,
    which can be loaded in subsequent runs to avoid re-processing features.

    Parameters
    ----------
    fingerprinter : WebsiteFingerprintModeler
        initialized fingerprinting engine
    pool : ProcessPool
        Pool to use for multiprocessing.
        Do not perform multiprocessing if None.
    checkpoint : str
        Path to ascii file to save individual leakage checkpoint information.
        Do not perform checkpointing if None.

    Returns
    -------
    list
        list of leakages where the index of each leakage maps to the feature number

    """
    leakage_indiv = []

    # open a checkpoint file
    if checkpoint:
        tmp_file = open(checkpoint, 'a+')
        past_leaks = [float(line) for line in tmp_file]
        lines = len(past_leaks)
        leakage_indiv = past_leaks

    # if a pool has been provided, perform computation in parallel
    # otherwise do serial computation
    if checkpoint:
        features = fingerprinter.data.features[lines:]
    else:
        features = fingerprinter.data.features
    if pool is None:
        proc_results = map(fingerprinter, features)
    else:
        proc_results = pool.imap(fingerprinter, features)
        pool.close()
    size = len(fingerprinter.data.features)  # number of features

    logger.info("Begin individual leakage measurements.")
    # measure information leakage
    # log current progress at twenty intervals
    for leakage in proc_results:
        if len(leakage_indiv) % int(size*0.05) == 0:
            logger.info("Progress: {}/{}".format(len(leakage_indiv), size))
        leakage_indiv.append(leakage)
        if checkpoint:
            tmp_file.write('{}\n'.format(str(leakage)))
            tmp_file.flush()
    if pool is not None:
        pool.join()
        pool.restart()
    if checkpoint:
        tmp_file.close()
    return leakage_indiv


def parse_args():
    """
    Parse command line arguments

    Accepted arguments:
      (f)eatures    -- directory which contains feature files
      (o)output     -- directory where to save analysis results
      n_samples     -- samples for montecarlo
      nmi_threshold -- redundancy threshold value
      topn          -- number of features to keep after pruning
      n_procs       -- number of processes to use for analysis

    Returns
    -------
    Namespace
        Argument namespace object

    """
    parser = argparse.ArgumentParser("TODO: Program description")

    # Required Arguments
    # directory containing feature files
    parser.add_argument("-f", "--features",
                        required=True,
                        type=str,
                        help="Directory which contains files with the .feature extension.")

    parser.add_argument("-o", "--output",
                        required=True,
                        type=str,
                        help="Directory location where to store analysis results.")

    # Optional Arguments
    # number of samples for monte-carlo integration
    parser.add_argument("--n_samples",
                        type=int,
                        default=5000,
                        help="The number of samples to use when performing Monte-Carlo Integration estimation. "
                             "Higher values result in more accurate measurements, but longer runtimes.")
    # redundancy threshold
    parser.add_argument("--nmi_threshold",
                        type=float,
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
                        default=0,
                        help="The number of processes to use when performing parallel operations. "
                             "Use '0' to use all available processors.")
    return parser.parse_args()


def main(args):
    """
    execute main logic
    """
    # prepare feature dataset
    logger.info("Loading dataset.")
    X, Y = load_data(args.features)
    feature_data = WebsiteData(X, Y)
    logger.info("Loaded {} sites.".format(len(feature_data.sites)))
    logger.info("Loaded {} instances.".format(len(feature_data)))

    # create process pool
    if args.n_procs > 1:
        pool = Pool(args.n_procs)
    elif args.n_procs == 0:
        pool = Pool(cpu_count())
    else:
        pool = None

    # directory to save results
    outdir = args.output
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # initialize fingerprint modeler
    fingerprinter = WebsiteFingerprintModeler(feature_data,
                                              sample_size=args.n_samples)

    # load previous leakage measurements if possible
    indiv_path = os.path.join(outdir, 'indiv.pkl')
    if os.path.exists(indiv_path):
        with open(indiv_path, "rb") as fi:
            logger.info("Loading saved individual leakage measures.")
            leakage_indiv = dill.load(fi)

    # otherwise do individual measure
    else:
        leakage_indiv = individual_measure(fingerprinter, pool,
                                           checkpoint=os.path.join(outdir, 'indiv_checkpoint.txt'))

        # save individual leakage to file
        logger.info("Saving individual leakage to {}.".format(indiv_path))
        with open(indiv_path, "wb") as fi:
            dill.dump(leakage_indiv, fi)

    # perform combined information leakage measurements
    # load joint leakage file
    comb_path = os.path.join(outdir, 'comb.pkl')
    if os.path.exists(comb_path):
        with open(args.combined, "rb") as fi:
            logger.info("Loading saved joint leakage measures.")
            leakage_joint = dill.load(fi)

    # otherwise do joint leakage estimation
    else:
        # initialize MI analyzer
        analyzer = MutualInformationAnalyzer(feature_data, pool=pool)

        # process into list of non-redundant features
        logger.info("Begin feature pruning.")
        cleaned, pruned = analyzer.prune(features=feature_data.features,
                                         leakage=leakage_indiv,
                                         nmi_threshold=args.nmi_threshold,
                                         topn=args.topn,
                                         checkpoint=os.path.join(outdir, 'prune_checkpoint.txt'))
        with open(os.path.join(outdir, 'top{}_cleaned.pkl'.format(args.topn)), 'wb') as fi:
            dill.dump(cleaned, fi)
        with open(os.path.join(outdir, 'top{}_redundant.pkl'.format(args.topn)), 'wb') as fi:
            dill.dump(pruned, fi)

        # cluster non-redundant features
        logger.info("Begin feature clustering.")
        clusters, distance_matrix = analyzer.cluster(cleaned)
        with open(os.path.join(outdir, 'distance_matrix.pkl'), 'w') as fi:
            dill.dump(distance_matrix, fi)
        with open(os.path.join(outdir, 'clusters.pkl'), 'w') as fi:
            dill.dump(clusters, fi)

        # perform joint information leakage measurement
        logger.info('Identified {} clusters.'.format(len(clusters)))
        logger.info("Begin cluster leakage measurements.")
        leakage_joint = fingerprinter.information_leakage(clusters)

        # save individual leakage to file
        logger.info("Saving joint leakage to {}.".format(args.combined))
        if os.path.dirname(args.combined):
            os.makedirs(os.path.dirname(args.combined))
        with open(comb_path, "wb") as fi:
            dill.dump(leakage_joint, fi)

    logger.info("Finished execution.")


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(-1)


