from features.common import X


def First20(times, sizes, features):
    for i in range(0, 20):
        try:
            features.append(sizes[i] + 1500)
        except:
            features.append(X)


def First30PktNum(times, sizes, features):
    out_count = 0
    in_count = 0
    for i in range(0, 30):
        # handle traces having less than 30 packets
        if i < len(sizes):
            if sizes[i] > 0:
                out_count = out_count + 1
            else:
                in_count = in_count + 1

    features.append(out_count)
    features.append(in_count)


def Last30PktNum(times, sizes, features):
    out_count = 0
    in_count = 0
    for i in range(1, 31):
        # handle traces having less than 30 packets
        if i <= len(sizes):
            if sizes[-i] > 0:
                out_count += 1
            else:
                in_count += 1
    features.append(out_count)
    features.append(in_count)
