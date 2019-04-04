# the feature extraction script
# include a complete list of features in the Tor website fingerprinting literature
from __future__ import division
import os
import util
import argparse
import multiprocessing
from preprocess.features import *
from util import FEATURE_EXT, BATCH_NUM, DEBUG_FLAG, NORMALIZE_TRAFFIC, PACKET_NUMBER, PKT_TIME, UNIQUE_PACKET_LENGTH, \
    NGRAM_ENABLE, TRANS_POSITION, PACKET_DISTRIBUTION, BURSTS, FIRST20, CUMUL, FIRST30_PKT_NUM, LAST30_PKT_NUM, \
    PKT_PER_SECOND, INTERVAL_KNN, INTERVAL_ICICS, INTERVAL_WPES11, howlong, featureCount


def create_file(dir_file, out_path):
    """
    create a celltrace file in corresponding folder
    return the path of the created file
    """
    filename = os.path.basename(dir_file)
    new_name = filename.split(".")[0] + FEATURE_EXT
    dest_path = os.path.join(out_path, new_name)

    # create folder if nonexist
    dest_dir = os.path.dirname(dest_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    os.mknod(dest_path)
    return dest_path


def enumerate_files(dir):
    """
    recursively enumerate files in a directory root
    """
    file_list = []
    for dirname, dirnames, filenames in os.walk(dir):
        # skip logs directory
        if "logs" in dirnames:
            dirnames.remove("logs")
        # if file exists
        if len(filenames) != 0:
            for filename in filenames:
                fulldir = os.path.join(dirname, filename)
                file_list.append(fulldir)
    return file_list


def extract(times, sizes, features, debug_path="./"):
    """
    extract features from a parsed website trace
    """
    feature_pos = dict()

    # Transmission size features
    if PACKET_NUMBER:
        PktNum.PacketNumFeature(times, sizes, features)
        if DEBUG_FLAG:
            feature_pos['PACKET_NUMBER'] = len(features)

    # inter packet time + transmission time feature
    if PKT_TIME:
        Time.TimeFeature(times, sizes, features)
        if DEBUG_FLAG:
            feature_pos['PKT_TIME'] = len(features)

    # Unique packet lengths
    if UNIQUE_PACKET_LENGTH:
        PktLen.PktLenFeature(times, sizes, features)
        if DEBUG_FLAG:
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
        if DEBUG_FLAG:
            feature_pos['NGRAM'] = len(features)

    # trans position features
    if TRANS_POSITION:
        TransPosition.TransPosFeature(times, sizes, features)
        if DEBUG_FLAG:
            feature_pos['TRANS_POSITION'] = len(features)

    if INTERVAL_KNN:
        Interval.IntervalFeature(times, sizes, features, 'KNN')
        if DEBUG_FLAG:
            feature_pos['INTERVAL_KNN'] = len(features)
    if INTERVAL_ICICS:
        Interval.IntervalFeature(times, sizes, features, 'ICICS')
        if DEBUG_FLAG:
            feature_pos['INTERVAL_ICICS'] = len(features)
    if INTERVAL_WPES11:
        Interval.IntervalFeature(times, sizes, features, 'WPES11')
        if DEBUG_FLAG:
            feature_pos['INTERVAL_WPES11'] = len(features)

    # Packet distributions (where are the outgoing packets concentrated) (knn + k-anonymity)
    if PACKET_DISTRIBUTION:
        PktDistribution.PktDistFeature(times, sizes, features)
        if DEBUG_FLAG:
            feature_pos['PKT_DISTRIBUTION'] = len(features)

    # Bursts (knn)
    if BURSTS:
        Burst.BurstFeature(times, sizes, features)
        if DEBUG_FLAG:
            feature_pos['BURST'] = len(features)

    # first 20 packets (knn)
    if FIRST20:
        HeadTail.First20(times, sizes, features)
        if DEBUG_FLAG:
            feature_pos['FIRST20'] = len(features)

    # first 30: outgoing/incoming packet number (k-anonymity)
    if FIRST30_PKT_NUM:
        HeadTail.First30PktNum(times, sizes, features)
        if DEBUG_FLAG:
            feature_pos['FIRST30_PKT_NUM'] = len(features)

    # last 30: outgoing/incoming packet number (k-anonymity)
    if LAST30_PKT_NUM:
        HeadTail.Last30PktNum(times, sizes, features)
        if DEBUG_FLAG:
            feature_pos['LAST30_PKT_NUM'] = len(features)

    # packets per second (k-anonymity)
    # plus alternative list
    if PKT_PER_SECOND:
        PktSec.PktSecFeature(times, sizes, features, howlong)
        if DEBUG_FLAG:
            feature_pos['PKT_PER_SECOND'] = len(features)

    # CUMUL features
    if CUMUL:
        features.extend(Cumul.CumulFeatures(sizes, featureCount))
        if DEBUG_FLAG:
            feature_pos['CUMUL'] = len(features)

    if DEBUG_FLAG:
        # output FeaturePos
        with open(os.path.join(debug_path, 'FeaturePos'), 'w') as fd:
            newfp = sorted(feature_pos.items(), key=lambda i: i[1])
            for each_key, pos in newfp:
                fd.write(each_key + ':' + str(pos) + '\n')


def batch_handler(file_list, out_path):
    """
    handle feature extraction for each trace instance assigned to batch
    """
    for filepath in file_list:
        if FEATURE_EXT in filepath:
            continue

        # load trace file
        times = []
        sizes = []
        with open(filepath, "r") as f:
            try:
                for x in f:
                    x = x.split("\t")
                    times.append(float(x[0]))
                    sizes.append(int(x[1]))
            except:
                pass

        # whether times or size is empty
        if len(times) == 0 or len(sizes) == 0:
            continue

        # whether normalize traffic
        if NORMALIZE_TRAFFIC == 1:
            times, sizes = util.normalize_traffic(times, sizes)

        features = []
        try:
            extract(times, sizes, features, debug_path=out_path)
        except:
            print("error occured:", filepath)
            continue

        dest = create_file(filepath, out_path)
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

    # split into BATCH_NUM files
    flist_batch = [[]] * BATCH_NUM
    for idx, each_file in enumerate_files(file_list):
        bdx = idx % BATCH_NUM
        flist_batch[bdx].append(each_file)

    # start BATCH_NUM processes for computation
    pjobs = []
    for i in range(BATCH_NUM):
        p = multiprocessing.Process(target=batch_handler, args=(flist_batch[i], out_path))
        pjobs.append(p)
        p.start()
    for eachp in pjobs:
        eachp.join()
    print("finished!")


def parse_args():
    """
    parse command line arguments
    """
    parser = argparse.ArgumentParser("Process traces into features lists.")
    parser.add_argument("-t", "--traces", required=False)
    parser.add_argument("-e", "--extension", required=False)
    parser.add_argument("-o", "--output", required=False)
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
