from math import sqrt


def avg(values):

    return sum(values)/len(values)

def sdev(values):

    mean = avg(values)
    sdev = sqrt(sum([abs(v - mean) ** 2 for v in values]) / len(values))

    return sdev


def zscores(values, lookback, scaling=1):

    result = []

    for i in range(len(values)):
        values_slice = values[i - lookback:i+1] if i >= lookback else values[:i+1]

        mean = avg(values_slice)
        sdev = sdev(values_slice)
        # sdev = sqrt(sum([abs(v-mean)**2 for v in values_slice])/len(values_slice))

        v = values[i]

        result.append(((v - mean)/sdev)*scaling if sdev != 0 else 0)

    return result

def min_max_norm(values, lookback, only_positive=True):
    result = []

    for i in range(len(values)):
        subset = None
        if i <= lookback:
            subset = values[0:i+1]
        else:
            subset = values[i-lookback:i+1]

        min_value = min(subset)
        max_value = max(subset)

        v = values[i]

        if max_value == min_value:
            result.append(0)
        else:
            result.append((v - min_value)/(max_value - min_value) if only_positive
                          else
                          2*(v - min_value)/(max_value - min_value)-1)

    return result

def get_label_distribution(labels, classes):

    return [len([l for l in labels if l == c])/len(labels) for c in classes]

def exponential_smoothing(values, alpha):
    result = []
    for i in range(len(values)):
        if i == 0:
            result.append(values[i])
        else:
            result.append(alpha*values[i]+(1-alpha)*result[i-1])
    return result