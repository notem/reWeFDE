"""
Main project file which performs info-leak measure
"""
import argparse
import numpy as np
import csv
import os
from fingerprint_modeler import WebsiteFingerprintModeler


def info_leakage(X, Y):
    """
    Compute individual information leakage of features
    """
    leakage = []
    fingerprinter = WebsiteFingerprintModeler(X, Y)
    for i in range(X.shape()[1]):
        leakage[i] = fingerprinter.individual_leakage(i)


def load_data(directory):
    """
    Load feature files from feature directory
    :return X - numpy array of data instances (shape {n,f})
    :return Y - numpy array of data labels (shape {n,1})
    """
    X = []
    Y = []
    for root, dirs, files in os.walk(directory):
        files = [fi for fi in files if fi.endswith(".features")]    # filter
        for file in files:
            cls, ins = file.split("-")
            with open(os.path.join(root, file), "r") as csvFile:
                features = np.array(list(csv.reader(csvFile))).squeeze()
                X.append(features)
                Y.append(int(cls))
    return np.array(X), np.array(Y)


def parse_args():
    """
    Parse command line arguments
    Accepted arguments:
      (t)races -- directory which contains feature files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--traces")
    return parser.parse_args()


def main():
    """
    Program entrypoint
    """
    args = parse_args()

    X, Y = load_data(args.traces)
    leakage = info_leakage(X, Y)

    print(leakage)
