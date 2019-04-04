# for CUMUL feature extractions
# input: a list of packet sizes


import itertools
import numpy


def CumulFeatures(packets, featureCount):
    separateClassifier = True
    # Calculate Features

    features = []

    total = []
    cum = []
    pos = []
    neg = []
    inSize = 0
    outSize = 0
    inCount = 0
    outCount = 0

    # Process trace
    for packetsize in itertools.islice(packets, None):

        # CUMUL uses positive to denote incoming, negative to be outgoing,
        # different from dataset
        packetsize = - packetsize

        # incoming packets
        if packetsize > 0:
            inSize += packetsize
            inCount += 1
            # cumulated packetsizes
            if len(cum) == 0:
                cum.append(packetsize)
                total.append(packetsize)
                pos.append(packetsize)
                neg.append(0)
            else:
                cum.append(cum[-1] + packetsize)
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + packetsize)
                neg.append(neg[-1] + 0)

        # outgoing packets
        if packetsize < 0:
            outSize += abs(packetsize)
            outCount += 1
            if len(cum) == 0:
                cum.append(packetsize)
                total.append(abs(packetsize))
                pos.append(0)
                neg.append(abs(packetsize))
            else:
                cum.append(cum[-1] + packetsize)
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + 0)
                neg.append(neg[-1] + abs(packetsize))

    # Should already be removed by outlier Removal
    # if len(cum) < 2:
    # something must be wrong with this capture
    # continue

    # add feature
    # features.append(classLabel)
    features.append(inCount)
    features.append(outCount)
    features.append(outSize)
    features.append(inSize)

    if separateClassifier:
        # cumulative in and out
        posFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount / 2), total, pos)
        negFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount / 2), total, neg)
        for el in itertools.islice(posFeatures, None):
            features.append(el)
        for el in itertools.islice(negFeatures, None):
            features.append(el)
    else:
        # cumulative in one
        cumFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount + 1), total, cum)
        for el in itertools.islice(cumFeatures, 1, None):
            features.append(el)

    return features
    # fdout.write(str(features[0]) + ' '  + ' '.join(['%d:%s' % (i+1, el) for i,el in enumerate(features[1:])]) + ' # ' + str(instance.timestamp) + '\n')
