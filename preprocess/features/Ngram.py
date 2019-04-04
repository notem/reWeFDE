def NgramLocator(sample, Ng):
    # locate which gram in NgramExtract
    index = 0
    for i in range(0, Ng):
        if sample[i] == 1:
            bit = 1
        else:
            bit = 0
        index = index + bit * (2 ** (Ng - i - 1))
    return index


def NgramExtract(sizes, NGRAM):
    # n-gram feature for ordering
    counter = 0
    buckets = [0] * (2 ** NGRAM)
    for i in range(0, len(sizes) - NGRAM + 1):
        index = NgramLocator(sizes[i:i + NGRAM], NGRAM)
        buckets[index] = buckets[index] + 1
        counter = counter + 1
    return buckets
