import numpy
from common import X


# Transpositions (similar to good distance scheme)
# how many packets are in front of the outgoing/incoming packet?
def TransPosFeature(times, sizes, features):
    # for outgoing packets
    count = 0
    temp = []
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            count += 1
            features.append(i)
            temp.append(i)
        if count == 300:
            break
    for i in range(count, 300):
        features.append(X)
    # std
    features.append(numpy.std(temp))
    # ave
    features.append(numpy.mean(temp))

    # for incoming packets
    count = 0
    temp = []
    for i in range(0, len(sizes)):
        if sizes[i] < 0:
            count += 1
            features.append(i)
            temp.append(i)
        if count == 300:
            break
    for i in range(count, 300):
        features.append(X)
    # std
    features.append(numpy.std(temp))
    # ave
    features.append(numpy.mean(temp))
