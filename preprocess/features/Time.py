import numpy
from features.common import X


# max, mean, std, quartile
def interTimeStats(times):
    res = []
    for i in range(1, len(times)):
        prev = times[i - 1]
        cur = times[i]
        res.append(cur - prev)

    if len(res) == 0:
        return [X, X, X, X]
    else:
        return [numpy.max(res), numpy.mean(res), numpy.std(res), numpy.percentile(res, 75)]


# transmission stats
# 25, 50, 75, 100 quartiles
def transTimeStats(times):
    return [numpy.percentile(times, 25),
            numpy.percentile(times, 50),
            numpy.percentile(times, 75),
            numpy.percentile(times, 100)]


# k-anonymity
# inter packet time statistics for total, incoming, and outgoing
# max, mean, std, third quartile
def TimeFeature(times, sizes, features):
    # inter packet time feature
    # total
    features.extend(interTimeStats(times))
    # outgoing
    times_out = []
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            times_out.append(times[i])
    features.extend(interTimeStats(times_out))

    # incoming
    times_in = []
    for i in range(0, len(sizes)):
        if sizes[i] < 0:
            times_in.append(times[i])
    features.extend(interTimeStats(times_in))

    # transmission time feature
    # total
    features.extend(transTimeStats(times))
    # outgoing
    features.extend(transTimeStats(times_out))
    # incoming
    features.extend(transTimeStats(times_in))
