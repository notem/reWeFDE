import argparse
import dill
import sys
from matplotlib import pyplot as plt


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    Namespace
        a namespace containing the parsed command line arguments

    """
    parser = argparse.ArgumentParser("Plot the results of mutual information analysis for the WeFDE technique.")
    parser.add_argument('-d', '--distances',
                        type=str, required=True)
    return parser.parse_args()


def main(distances_file):
    """
    Load the distance matrix and show plot.

    Parameters
    ----------
    distances_file : str
        Path to distances pickle file.

    """
    with open(distances_file, 'rb') as fi:
        distances = list(dill.load(fi))
    plt.matshow(distances, cmap='Blues_r')
    plt.show()


if __name__ == "__main__":
    try:
        args = parse_args()
        main(args.distances)
    except KeyboardInterrupt:
        sys.exit(-1)

