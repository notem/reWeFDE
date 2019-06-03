import argparse
import os
import multiprocessing
from info_leakage.preprocess.extract import enumerate_files, BATCH_NUM, batch_handler


def main(trace_path, out_path):
    """
    start batches to handle feature extraction
    """
    file_list = enumerate_files(trace_path)

    # split into BATCH_NUM files
    flist_batch = [[]] * BATCH_NUM
    for idx, each_file in enumerate(file_list):
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