import numpy as np
import matplotlib.pyplot as plt
import dill
import argparse
import sys
from common import FEATURE_CATEGORIES, COLORS

plt.rcParams["font.family"] = "serif"
plt.gcf().subplots_adjust(left=0.25)
plt.gcf().subplots_adjust(bottom=0.1)


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    Namespace
        a namespace containing the parsed command line arguments

    """
    parser = argparse.ArgumentParser("Plot the results of mutual information analysis for the WeFDE technique.")
    parser.add_argument('-c', '--cluster_file',
                        type=str, required=True)
    parser.add_argument('-r', '--redundant_file',
                        type=str, required=False, default=None)
    parser.add_argument('-s', '--style',
                        type=str, required=False,
                        default='horizontal', choices=['horizontal', 'vertical'])
    return parser.parse_args()


def main(cluster_file, redundant_file=None, style='horizontal'):
    """
    Produce a stacked bar graph demonstrating the results of feature clustering.

    Parameters
    ----------
    cluster_file : str
        pickle file containing feature clusters
    redundant_file : str
        pickle file containing features which were pruned
    style : str
        the bar graph style (ie. horizontal or vertical bars)

    """
    with open(cluster_file, 'rb') as fi:
        clusters = list(dill.load(fi))
    if redundant_file:
        with open(redundant_file, 'rb') as fi:
            redundant = list(dill.load(fi))

    # clusters with only one feature are 'noise'
    # group them into one 'noise' cluster for display
    noise_cluster = []
    real_clusters = []
    clustered = 0
    for cluster in clusters:
        if len(cluster) == 1:
            noise_cluster.extend(cluster)
        else:
            real_clusters.append(cluster)
            clustered += len(cluster)
    clusters = real_clusters

    # determine total count of features for each category
    total_counts = [0 for _ in FEATURE_CATEGORIES]
    for i, cluster in enumerate(clusters):
        for feature in cluster:
            for j, cat in enumerate(FEATURE_CATEGORIES):
                if feature+1 >= cat[1][0] and feature <= cat[1][1]:
                    total_counts[j] += 1
    for feature in noise_cluster:
        for j, cat in enumerate(FEATURE_CATEGORIES):
            if feature+1 >= cat[1][0] and feature <= cat[1][1]:
                total_counts[j] += 1
    if redundant_file:
        for feature in redundant:
            for j, cat in enumerate(FEATURE_CATEGORIES):
                if feature+1 >= cat[1][0] and feature <= cat[1][1]:
                    total_counts[j] += 1

    plots = []
    cluster_names = []
    category_names = list(zip(*FEATURE_CATEGORIES))[0]
    plot_indices = np.arange(len(category_names))

    def plot_cluster(cluster, color, name, style='horizontal'):
        """
        Plot a cluster on the figure.

        Note
        ----
        This function appends items to the ``cluster_names`` list (defined in outer scope).
        The ``total_counts`` list is also modified as a result of calling this function.

        Parameters
        ----------
        cluster : list
        color : str
        name : str
        style : str

        """
        cluster_names.append(name)

        # find the number of features of each category which belongs to the current cluster
        counts = [0 for _ in FEATURE_CATEGORIES]
        for feature in cluster:
            for j, cat in enumerate(FEATURE_CATEGORIES):
                if feature+1 >= cat[1][0] and feature <= cat[1][1]:
                    counts[j] += 1

        # plot the category bars
        if style == 'horizontal':
            p = plt.barh(plot_indices, total_counts, color=color)
        elif style == 'vertical':
            p = plt.bar(plot_indices, total_counts, color=color)
        else:
            print("Invalid plot style: \'{}\'".format(style))
            sys.exit(-1)
        plots.append(p)

        # subtract the count of
        for j in range(len(counts)):
            total_counts[j] -= counts[j]

    # plot each cluster
    if redundant_file:
        plot_cluster(redundant, 'grey', 'redundant', style)
    plot_cluster(noise_cluster, 'darkgrey', 'no cluster', style)
    for i, cluster in enumerate(clusters):
        plot_cluster(cluster, COLORS[i], 'cluster {}'.format(i), style)

    # verify that all features were plotted
    for count in total_counts:
        assert(count == 0)

    # define plot labels and legends
    #plt.title('Feature Clustering Results')
    label1 = 'Feature Count'
    label2 = 'Feature Category'
    if style == 'horizontal':
        plt.xlabel(label1, fontsize=14, fontweight='bold')
        plt.ylabel(label2, fontsize=14, fontweight='bold')
        plt.yticks(plot_indices, category_names, fontsize=8)

    elif style == 'vertical':
        plt.ylabel(label1, fontsize=14, fontweight='bold')
        plt.xlabel(label2, fontsize=14, fontweight='bold')
        plt.xticks(plot_indices, category_names, fontsize=8)

    plt.legend((p[0] for p in plots), cluster_names)

    plt.show()


if __name__ == "__main__":
    try:
        args = parse_args()
        main(args.cluster_file, args.redundant_file, args.style)
    except KeyboardInterrupt:
        sys.exit(-1)

