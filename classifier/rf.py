import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json
import sys
import os
import csv

#### Parameters ####
num_Trees = 1000
SEED = 1


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
                features = features[:13] + features[37:2813] + features[2939:]  # remove time features
                X.append(features)
                Y.append(int(cls))

    return np.array(X), np.array(Y)


def top_n_accuracy(preds, truths, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i in range(truths.shape[0]):
        if truths[i] in best_n[i, :]:
            successes += 1
    return float(successes)/truths.shape[0]


def load_features(path_to_features, tr_split):
    """
    Prepare monitored data for training and test sets.
    """

    # load features dataset
    #X_tr, Y_tr = load_data(os.path.join(path_to_features, 'train'), ".features", " ")
    #X_ts, Y_ts = load_data(os.path.join(path_to_features, 'test'), ".features", " ")

    X, Y = load_data(path_to_features, ".features", " ")
    # shuffle features
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, Y = X[s], Y[s]

    # split into training and testing
    cut = int(tr_split*Y.shape[0])
    X_tr, Y_tr, X_ts, Y_ts = X[:cut], Y[:cut], X[cut:], Y[cut:]

    return X_tr, Y_tr, X_ts, Y_ts


def classify(feature_directory, tr_split, out):
    """
    Closed world RF classification of data
    - only uses sk.learn classification - does not do additional k-nn.
    """

    # load dataset
    X_tr, Y_tr, X_ts, Y_ts = load_features(feature_directory, tr_split)

    #
    # train random forest model
    #
    print("Training ...")
    model = RandomForestClassifier(n_jobs=2,
                                   n_estimators=num_Trees,
                                   oob_score=True)
    model.fit(X_tr, Y_tr)

    #
    # test performance
    #
    acc = model.score(X_ts, Y_ts)
    print("accuracy = ", acc)

    pred = model.predict_proba(X_ts)
    acc_2 = top_n_accuracy(pred, Y_ts, 2)
    print("top_2 accuracy = ", acc_2)

    #
    # rank feature importance
    #
    print("Top 100 features:")
    importance = zip(model.feature_importances_,
                     range(0, len(model.feature_importances_)))
    sorted_importance = sorted(importance, key=lambda tup: tup[0], reverse=True)
    index = 0
    for score, label in sorted_importance:
        index += 1
        print("\t%d. Feature #%s (%f)" % (index, label, score))
        if index == 100:
            break

    #
    # cross validation score
    #
    #scores = cross_val_score(model, np.array(X_tr), np.array(Y_tr))
    #print("cross_val_score = ", scores.mean())
    #print("OOB score = ", model.oob_score_)

    if out:
        res = dict()
        #res['cross_val_score'] = scores.mean()
        res['oob'] = model.oob_score_
        res['accuracy'] = acc
        #res['feature_rank'] = sorted_importance
        with open(out, "w") as fi:
            json.dump(res, fi, sort_keys=True, indent=4, separators=(',', ': '))


def parse_arguments():
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description='RF benchmarks')
    parser.add_argument('-f', '--features',
                        type=str,
                        help="Path to feature dictionary.",
                        required=True)
    parser.add_argument('-t', '--train',
                        default=0.8,
                        type=float,
                        help="Percentage of dataset to use for training.",
                        required=False)
    parser.add_argument('-o', '--output',
                        default=None,
                        help="Output file to store results.")
    return parser.parse_args()


def main():
    """
    Run RF classification using WeFDE features.
    """
    args = parse_arguments()

    # Example command line:
    # $ python rf.py --features /path/to/features --train 0.8
    classify(args.features, args.train, args.output)

    return 0


if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
