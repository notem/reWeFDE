import numpy


def PktSecFeature(times, sizes, features, howlong):
    count = [0] * howlong
    for i in range(0, len(sizes)):
        t = int(numpy.floor(times[i]))
        if t < howlong:
            count[t] = count[t] + 1
    features.extend(count)

    # mean, standard deviation, min, max, median
    features.append(numpy.mean(count))
    features.append(numpy.std(count))
    features.append(numpy.min(count))
    features.append(numpy.max(count))
    features.append(numpy.median(count))

    # alternative: 20 buckets
    bucket_num = 20
    bucket = [0] * bucket_num
    for i in range(0, howlong):
        ib = i // (howlong // bucket_num)
        bucket[ib] = bucket[ib] + count[i]
    features.extend(bucket)
    features.append(numpy.sum(bucket))
