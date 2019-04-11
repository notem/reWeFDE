"""
Main project file which performs info-leak measure
"""
import argparse
import numpy as np
import csv
import os
import sys
import pickle
from fingerprint_modeler import WebsiteFingerprintModeler


def info_leakage(X, Y):
    """
    Compute individual information leakage of features
    """
    leakage = []
    fingerprinter = WebsiteFingerprintModeler(X, Y)
    for i in range(X.shape[1]):
        print("Processing leakage for feature #{}...".format(i))
        try:
            leakage.append(fingerprinter.individual_leakage_multi(i))
        except KeyboardInterrupt:
            sys.exit(-1)
        except Exception as exc:
            print("=> Failed to estimate leakage: {}".format(exc))
            leakage.append(None)
    return leakage


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

    return np.array(X), np.array(Y)


def parse_args():
    """
    Parse command line arguments
    Accepted arguments:
      (t)races -- directory which contains feature files
    """
    parser = argparse.ArgumentParser("")
    parser.add_argument("-t", "--traces")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Program entrypoint
    """
    args = parse_args()

    X, Y = load_data(args.traces)
    leakage = info_leakage(X, Y)
    pickle.dump(leakage, open("leakage.pkl", "wb"))
    print(leakage)
