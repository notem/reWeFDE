from common import X

# knn feature (share similarity with interval)
# the burst of inflow traffic
def BurstFeature(times, sizes, features):
    bursts = []
    curburst = 0
    stopped = 0
    for x in sizes:
        # find two bugs in original wang's
        # two adjacent... should be nested "if"
        # burst of outgoing packets, not incoming ones

        if x > 0:
            stopped = 0
            curburst += x
        if x < 0 and stopped == 0:
            stopped = 1
        elif x < 0 and stopped == 1:
            stopped = 0
            if curburst != 0:
                bursts.append(curburst)
            curburst = 0  # corrected a bug: reset curburst!!
        else:
            pass

    # burst could be none
    if len(bursts) != 0:
        features.append(max(bursts))
        features.append(sum(bursts) / len(bursts))
        features.append(len(bursts))
    else:
        features.append(0)
        features.append(0)
        features.append(0)

    counts = [0, 0, 0]
    for x in bursts:
        if x > 5:
            counts[0] += 1
        if x > 10:
            counts[1] += 1
        if x > 15:
            counts[2] += 1

    features.append(counts[0])
    features.append(counts[1])
    features.append(counts[2])
    for i in range(0, 5):
        try:
            features.append(bursts[i])
        except:
            features.append(X)
