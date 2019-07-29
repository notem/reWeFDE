
# extract params
FEATURE_EXT = ".features"
NORMALIZE_TRAFFIC = 0

# turn on/off debug:
PACKET_NUMBER = True
PKT_TIME = True
UNIQUE_PACKET_LENGTH = False
NGRAM_ENABLE = True
TRANS_POSITION = True
PACKET_DISTRIBUTION = True
BURSTS = True
FIRST20 = True
CUMUL = True
FIRST30_PKT_NUM = True
LAST30_PKT_NUM = True
PKT_PER_SECOND = True
INTERVAL_KNN = True
INTERVAL_ICICS = True
INTERVAL_WPES11 = True

# packet number per second, how many seconds to count?
howlong = 100

# n-gram feature
NGRAM = 3

# CUMUL feature number
featureCount = 100


# Python3 conversion of python2 cmp function
def cmp(a, b):
    return (a > b) - (a < b)


# normalize traffic
def normalize_traffic(times, sizes):
    # sort
    tmp = sorted(zip(times, sizes))

    times = [x for x, _ in tmp]
    sizes = [x for _, x in tmp]

    TimeStart = times[0]
    PktSize = 500

    # normalize time
    for i in range(len(times)):
        times[i] = times[i] - TimeStart

    # normalize size
    for i in range(len(sizes)):
        sizes[i] = (abs(sizes[i]) / PktSize) * cmp(sizes[i], 0)

    # flat it
    newtimes = list()
    newsizes = list()

    for t, s in zip(times, sizes):
        numCell = abs(s)
        oneCell = cmp(s, 0)
        for r in range(numCell):
            newtimes.append(t)
            newsizes.append(oneCell)

    return newtimes, newsizes

