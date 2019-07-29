# the feature extraction script
# include a complete list of features in the Tor website fingerprinting literature
from __future__ import division
import os
import argparse
import sys
import itertools
import json
from tqdm import tqdm
from multiprocessing import Pool
from collections import OrderedDict
import re

import util
from features import *
from util import FEATURE_EXT, NORMALIZE_TRAFFIC, PACKET_NUMBER, PKT_TIME, UNIQUE_PACKET_LENGTH, \
    NGRAM_ENABLE, TRANS_POSITION, PACKET_DISTRIBUTION, BURSTS, FIRST20, CUMUL, FIRST30_PKT_NUM, LAST30_PKT_NUM, \
    PKT_PER_SECOND, INTERVAL_KNN, INTERVAL_ICICS, INTERVAL_WPES11, howlong, featureCount


def enumerate_files(dir, splitter='-', extension=''):
    """
    recursively enumerate files in a directory root
    """
    file_list = []
    for dirname, dirnames, filenames in os.walk(dir):
        # filter out invalid file names
        filenames = [filename for filename in filenames
                     if re.fullmatch('\\d+{}\\d+{}'.format(splitter, extension), filename)]
        for filename in filenames:
            file_list.append(os.path.join(dirname, filename))
    return file_list


def extract(times, sizes, debug_path="./", store_feature_pos=False):
    """
    extract features from a parsed website trace
    """
    feature_pos = OrderedDict()
    features = []

    # Transmission size features
    if PACKET_NUMBER:
        PktNum.PacketNumFeature(times, sizes, features)
        if store_feature_pos:
            feature_pos['PACKET_NUMBER'] = len(features)

    # inter packet time + transmission time feature
    if PKT_TIME:
        Time.TimeFeature(times, sizes, features)
        if store_feature_pos:
            feature_pos['PKT_TIME'] = len(features)

    # Unique packet lengths
    if UNIQUE_PACKET_LENGTH:
        PktLen.PktLenFeature(times, sizes, features)
        if store_feature_pos:
            feature_pos['UNIQUE_PACKET_LENGTH'] = len(features)

    # n-gram feature for ordering
    if NGRAM_ENABLE:
        buckets = Ngram.NgramExtract(sizes, 2)
        features.extend(buckets)
        buckets = Ngram.NgramExtract(sizes, 3)
        features.extend(buckets)
        buckets = Ngram.NgramExtract(sizes, 4)
        features.extend(buckets)
        buckets = Ngram.NgramExtract(sizes, 5)
        features.extend(buckets)
        buckets = Ngram.NgramExtract(sizes, 6)
        features.extend(buckets)
        if store_feature_pos:
            feature_pos['NGRAM'] = len(features)

    # trans position features
    if TRANS_POSITION:
        TransPosition.TransPosFeature(times, sizes, features)
        if store_feature_pos:
            feature_pos['TRANS_POSITION'] = len(features)

    if INTERVAL_KNN:
        Interval.IntervalFeature(times, sizes, features, 'KNN')
        if store_feature_pos:
            feature_pos['INTERVAL_KNN'] = len(features)
    if INTERVAL_ICICS:
        Interval.IntervalFeature(times, sizes, features, 'ICICS')
        if store_feature_pos:
            feature_pos['INTERVAL_ICICS'] = len(features)
    if INTERVAL_WPES11:
        Interval.IntervalFeature(times, sizes, features, 'WPES11')
        if store_feature_pos:
            feature_pos['INTERVAL_WPES11'] = len(features)

    # Packet distributions (where are the outgoing packets concentrated) (knn + k-anonymity)
    if PACKET_DISTRIBUTION:
        PktDistribution.PktDistFeature(times, sizes, features)
        if store_feature_pos:
            feature_pos['PKT_DISTRIBUTION'] = len(features)

    # Bursts (knn)
    if BURSTS:
        Burst.BurstFeature(times, sizes, features)
        if store_feature_pos:
            feature_pos['BURST'] = len(features)

    # first 20 packets (knn)
    if FIRST20:
        HeadTail.First20(times, sizes, features)
        if store_feature_pos:
            feature_pos['FIRST20'] = len(features)

    # first 30: outgoing/incoming packet number (k-anonymity)
    if FIRST30_PKT_NUM:
        HeadTail.First30PktNum(times, sizes, features)
        if store_feature_pos:
            feature_pos['FIRST30_PKT_NUM'] = len(features)

    # last 30: outgoing/incoming packet number (k-anonymity)
    if LAST30_PKT_NUM:
        HeadTail.Last30PktNum(times, sizes, features)
        if store_feature_pos:
            feature_pos['LAST30_PKT_NUM'] = len(features)

    # packets per second (k-anonymity)
    # plus alternative list
    if PKT_PER_SECOND:
        PktSec.PktSecFeature(times, sizes, features, howlong)
        if store_feature_pos:
            feature_pos['PKT_PER_SECOND'] = len(features)

    # CUMUL features
    if CUMUL:
        features.extend(Cumul.CumulFeatures(sizes, featureCount))
        if store_feature_pos:
            feature_pos['CUMUL'] = len(features)

    if store_feature_pos:
        # output FeaturePos
        with open(os.path.join(debug_path, 'FeaturePositions.json'), 'w') as fd:
            fd.write(json.dumps(feature_pos))

    return features


def task_handler(args):
    """
    handle feature extraction for each trace instance assigned to batch
    """
    filepath, out_path = args

    # load trace file
    times = []
    sizes = []
    with open(filepath, "r") as f:
        try:
            for x in f:
                x = x.split("\t")
                times.append(float(x[0]))
                sizes.append(int(x[1]))
        except KeyboardInterrupt:
            sys.exit(-1)
        except:
            return

    # whether normalize traffic
    if NORMALIZE_TRAFFIC == 1:
        times, sizes = util.normalize_traffic(times, sizes)

    # extract features (saving feature positions only for the first trace)
    features = extract(times, sizes,
                       debug_path=out_path,
                       store_feature_pos=os.path.basename(filepath) == '0-0')

    # save features to file
    dest = os.path.join(out_path, os.path.basename(filepath) + FEATURE_EXT)
    with open(dest, "w") as fout:
        for x in features:
            if isinstance(x, str):
                if '\n' in x:
                    fout.write(x)
                else:
                    fout.write(x + " ")
            else:
                fout.write(repr(x) + " ")


def main(trace_path, out_path):
    """
    start batches to handle feature extraction
    """
    file_list = enumerate_files(trace_path)

    # start BATCH_NUM processes for computation
    pool = Pool()
    for _ in tqdm(pool.imap(task_handler, zip(file_list, itertools.repeat(out_path))), total=len(file_list)):
        pass


def parse_args():
    """
    parse command line arguments
    """
    parser = argparse.ArgumentParser("Process traces into features lists.")
    parser.add_argument("-t", "--traces", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--extension", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.extension:
        FEATURE_EXT = args.extension
    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        main(args.traces, args.output)
    else:
        main(args.traces, args.traces)
