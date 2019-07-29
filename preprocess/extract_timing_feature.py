from util import howlong
from features import Time, PktSec
import os
import numpy as np


def extract_bursts(trace):
    bursts = []
    direction_counts = []
    direction = trace[0][1]
    burst = []
    count = 1
    for i, packet in enumerate(trace):
        if packet[1] != direction:
            bursts.append(burst)
            burst = [packet]
            direction_counts.append(count)
            direction = packet[1]
            count = 1
        else:
            burst.append(packet)
            count += 1
    bursts.append(burst)
    return bursts, direction_counts


def direction_counts(trace):
    counts = []
    direction = trace[0][1]
    count = 1
    for packet in trace:
        if packet[1] != direction:
            counts.append(count)
            direction = packet[1]
            count = 1
        else:
            count += 1
    return counts


def get_bin_sizes(feature_values, bin_input):
    bin_raw = []
    for v in feature_values.values():
        bin_raw.extend(v)
    bin_s = np.sort(bin_raw)
    bins = np.arange(0, 100 + 1, 100.0 / bin_input)

    final_bin = [np.percentile(bin_s, e) for e in bins]
    return final_bin


def slice_by_binsize(feature_values, bin_input):
    bin_for_all_instances = np.array(get_bin_sizes(feature_values, bin_input))
    d_new = {}
    for name, v in feature_values.iteritems():
        d_new[name] = [[] for _ in range(bin_input)]

        bin_indices = np.digitize(np.array(v),
                                  bin_for_all_instances[:bin_input],
                                  right=True)
        for i in range(bin_indices.size):
            if bin_indices[i] > bin_input:
                d_new[name][-1].append(v[i])
            elif bin_indices[i] == 0:
                d_new[name][0].append(v[i])
            else:
                d_new[name][bin_indices[i] - 1].append(v[i])
    return d_new


def get_statistics(feature_values, bin_input):
    sliced_dic = slice_by_binsize(feature_values, bin_input)
    bin_length = {
        key: [len(value) for value in values] for key, values in
        sliced_dic.iteritems()
    }
    return bin_length


def normalize_data(feature_values, bin_input):
    to_be_norm = get_statistics(feature_values, bin_input)
    normed = {
        key: [float(value) / sum(values) for value in values]
        if sum(values) > 0 else values
        for key, values in to_be_norm.iteritems()
    }
    return normed


def final_format_by_class(feature_values, bin_input):
    # norm_data = normalized_data(traces, bin_input)
    norm_data = get_statistics(feature_values, bin_input)
    final = {}
    for k in norm_data:
        c = k.split('-')[0]
        if c not in final:
            final[c] = [norm_data[k]]
        else:
            final[c].append(norm_data[k])
    return final


def padding_neural(feature_values):
    directed_neural = feature_values
    max_length = max(len(elements) for elements in directed_neural.values())
    print("Maximum Length", max_length)
    for key, value in directed_neural.iteritems():
        if len(value) < max_length:
            zeroes_needed = max_length - len(value)
            value += [0] * zeroes_needed

    return directed_neural

def intraBD_med(bursts):
    intra_burst_delays = []
    for burst in bursts:
        timestamps = [packet[0] for packet in burst]
        intra_burst_delays.append(np.median(timestamps))
    return intra_burst_delays


def inter_inramd(bursts):
    primary = intraBD_med(bursts)
    processed = [q-p for p, q in zip(primary[:-1], primary[1:])]

    return processed


def intra_burst_delay_var(bursts):
    intra_burst_delays = []
    for burst in bursts:
        timestamps = [packet[0] for packet in burst]
        intra_burst_delays.append(np.var(timestamps))
    return intra_burst_delays


def inter_burst_delay_first_first(bursts):
    timestamps = [float(burst[0][0]) for burst in bursts]

    return np.diff(timestamps).tolist()


def inter_burst_delay_incoming_first_first(bursts):
    incoming_bursts = [burst for burst in bursts if burst[0][1] == -1]
    timestamps = [float(burst[0][0]) for burst in incoming_bursts]
    return np.diff(timestamps).tolist()


def inter_burst_delay_last_first(bursts):
    timestamps_first = [float(burst[0][0]) for burst in bursts]
    timestamps_last = [float(burst[-1][0]) for burst in bursts]
    inter_burst_delays = [i-j for i, j in zip(timestamps_last,
                                              timestamps_first)]
    return inter_burst_delays


def inter_burst_delay_outgoing_first_first(bursts):
    outgoing_bursts = [burst for burst in bursts if burst[0][1] == 1]
    timestamps = [float(burst[0][0]) for burst in outgoing_bursts]
    return np.diff(timestamps).tolist()


def intra_interval(bursts):
    timestamps_first = [float(burst[0][0]) for burst in bursts]
    timestamps_last = [float(burst[-1][0]) for burst in bursts]
    interval = [i-j for i, j in zip(timestamps_last, timestamps_first)]
    return interval


def parse_args():
    """
    parse command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser("Process traces into features lists.")
    parser.add_argument("-t", "--traces", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-b", "--bin_size", default=20)
    parser.add_argument("-i", "--instances", default=1000)
    parser.add_argument("-s", "--sites", default=95)
    return parser.parse_args()

def main():
    features = {
        "medians": {},
        "ibdff": {},
        "ibdiff": {},
        "ibdlf": {},
        "ibdoff": {},
        "interval": {},
        "inter_inramd": {},
        "ibdbvar": {},
    }

    args = parse_args()

    data_path = args.traces
    num_sites = args.sites
    num_instances = args.instances
    bin_size = args.bin_size

    labels_instances = []
    old_timing_features = dict()
    feature_pos = dict()

    for site in range(0, num_sites):
        for label in range(0, num_instances):
            file_name = str(site) + "-" + str(label)
            # Directory of the raw data
            with open(os.path.join(data_path, file_name), "r") as file_pt:
                # load trace
                trace = []
                times = []
                sizes = []
                for line in file_pt:
                    x = line.strip().split('\t')
                    time = float(x[0])
                    size = 1 if float(x[1]) > 0 else -1
                    times.append(time)
                    sizes.append(size)
                    trace.append((time, size))

                # calculate and save k-FP timing features for the instance
                old_timing_features[file_name] = []
                Time.TimeFeature(times, sizes, old_timing_features[file_name])
                feature_pos['PKT_TIME'] = len(old_timing_features[file_name])
                PktSec.PktSecFeature(times, sizes, old_timing_features[file_name], howlong)
                feature_pos['PKT_PER_SEC'] = len(old_timing_features[file_name])

                # extract bursts and compute new timing statistics
                bursts, direction_counts = extract_bursts(trace)
                features["medians"][file_name] = intraBD_med(bursts)
                features["ibdff"][file_name] = \
                    inter_burst_delay_first_first(bursts)
                features["ibdiff"][file_name] = \
                    inter_burst_delay_incoming_first_first(bursts)
                features["ibdlf"][file_name] = \
                    inter_burst_delay_last_first(bursts)
                features["ibdoff"][file_name] = \
                    inter_burst_delay_outgoing_first_first(bursts)
                features["interval"][file_name] = intra_interval(bursts)
                features["inter_inramd"][file_name] = inter_inramd(bursts)
                features["ibdbvar"][file_name] = intra_burst_delay_var(bursts)
                labels_instances.append(file_name)
        print ("Done with Site: ", site)

    feature_bins = {
        "medians": bin_size,
        "ibdff": bin_size,
        "ibdiff": bin_size,
        "ibdlf": bin_size,
        "ibdoff": bin_size,
        "interval": bin_size,
        "inter_inramd": bin_size,
        "ibdbvar": bin_size
    }

    # Create bins for each feature, extract bin counts and normalize them
    print("Extracting Features...")

    for feature in features:
        features[feature] = normalize_data(features[feature], feature_bins[feature])

    feature_names = features.keys()

    output_dir = args.output

    print("Saving features...")
    for label in labels_instances:
        data = old_timing_features[label]
        for feature in feature_names:
            data.extend(features[feature][label])
            feature_pos[feature.upper()] = len(data)
        with open(os.path.join(output_dir, "{}.features".format(label)), "w") as out:
            out.write(' '.join([str(item) for item in data]))

    with open(os.path.join(output_dir, 'FeaturePos'), 'w') as fd:
        newfp = sorted(feature_pos.items(), key=lambda i: i[1])
        for each_key, pos in newfp:
            fd.write(each_key + ':' + str(pos) + '\n')

    print("Done")


if __name__ == '__main__':
    main()
