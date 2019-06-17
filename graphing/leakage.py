import math
import matplotlib.pyplot as plt
import dill
import argparse
import sys
from common import FEATURE_CATEGORIES, COLORS

# set figure font
plt.rcParams["font.family"] = "serif"

# define subplot grid dimensions
ROWS = 3
COLS = math.ceil(float(len(FEATURE_CATEGORIES)) / ROWS)


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    Namespace
        a namespace containing the parsed command line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', action='append')
    parser.add_argument('-n', '--name', action='append')

    args = parser.parse_args()
    assert(len(args.file) == len(args.name))
    return args


def main(leakage_files):
    """

    Parameters
    ----------
    leakage_files : list

    """
    # read independent leakage information from files
    leakages = []
    for path, _ in leakage_files:
        with open(path, 'rb') as fi:
            leakages.append(dill.load(fi))

    zipped_leakages = list(zip(*leakages))

    # start figure plot
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    for i in range(1, len(FEATURE_CATEGORIES) + 1):

        category, indices = FEATURE_CATEGORIES[i - 1]

        ax = fig.add_subplot(ROWS, COLS, i)
        ax.set_ylim(0, 4)
        ax.set_xticks([indices[0], indices[1]])
        ax.set_yticks(range(0, 5))
        ax.yaxis.grid(True, linestyle='dotted')
        ax.set_title(category, fontsize=11)

        # plot category leakages for each leakage file
        for j in range(len(leakage_files)):
            x = range(indices[0], indices[1]+1)
            slice = zipped_leakages[indices[0]-1: indices[1]]
            y = list(zip(*slice))[j]
            assert(len(x) == len(y))
            ax.plot(x, y, color=COLORS[j])

    fig.text(0.085, 0.5, 'Information Leakage (bit)',
             ha='center', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.03, 'Feature Category',
             ha='center', va='center', fontsize=16, fontweight='bold')
    plt.figlegend(labels=list(zip(*leakage_files))[1], loc='upper center', ncol=2)
    plt.show()


if __name__ == "__main__":
    try:
        args = parse_args()
        file_name_tuples = list(zip(args.file, args.name))
        main(file_name_tuples)
    except KeyboardInterrupt:
        sys.exit(-1)
