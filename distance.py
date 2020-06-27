import math


def l1(data1, data2):
    distance = 0

    for i1, i2 in data1, data2:
        distance += abs(i1 - i2)

    return distance


def l2(data1, data2):
    distance = 0

    for i1, i2 in data1, data2:
        r = i1 - i2
        distance += r * r

    return math.sqrt(distance)


def loss():
    pass