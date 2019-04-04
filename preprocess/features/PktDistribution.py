import numpy


def PktDistFeature(times, sizes, features):
    count = 0
    temp = []
    for i in range(0, min(len(sizes), 6000)):
        if sizes[i] > 0:
            count += 1
        if i % 30 == 29:
            features.append(count)
            temp.append(count)
            count = 0
    for i in range(len(sizes) // 30, 200):
        features.append(0)
        temp.append(0)
    # std
    features.append(numpy.std(temp))
    # mean
    features.append(numpy.mean(temp))
    # median
    features.append(numpy.median(temp))
    # max
    features.append(numpy.max(temp))

    # alternative packet distribution list (k-anonymity)
    # could be considered packet distributions with larger intervals
    num_bucket = 20
    bucket = [0] * num_bucket
    for i in range(0, 200):
        ib = i // (200 // num_bucket)
        bucket[ib] = bucket[ib] + temp[i]
    features.extend(bucket)
    features.append(numpy.sum(bucket))
