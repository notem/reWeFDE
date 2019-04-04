def roundArbitrary(x, base):
    return int(base * round(float(x) / base))


# packet number features
def PacketNumFeature(times, sizes, features):
    total = len(times)
    features.append(total)

    # count is outgoing pkt. number
    count = 0
    for x in sizes:
        if x > 0:
            count += 1
    features.append(count)
    features.append(total - count)

    # kanonymity also include incoming/total, outgoing/total
    out_total = float(count) / total
    in_total = float(total - count) / total
    features.append(out_total * 100)
    features.append(in_total * 100)

    # rounded version, from WPES 2011
    features.append(roundArbitrary(total, 15))
    features.append(roundArbitrary(count, 15))
    features.append(roundArbitrary(total - count, 15))

    features.append(roundArbitrary(out_total * 100, 5))
    features.append(roundArbitrary(in_total * 100, 5))

    # packet size in total (or called bandwidth)
    # should be the same with packet number, but anyway, include them

    features.append(total * 512)
    features.append(count * 512)
    features.append((total - count) * 512)
