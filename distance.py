import math


# Manhattan distance
def l1(data1, data2):
    distance = 0

    for i1, i2 in data1, data2:
        distance += abs(i1 - i2)

    return distance


# Euclidean distance
def l2(data1, data2):
    distance = 0

    for i1, i2 in data1, data2:
        diff = i1 - i2
        distance += diff * diff

    return math.sqrt(distance)


# data should be without of correct label
def loss(data, w_of_correct_label):
    L = 0

    for w_of_labels in data:
        L += max(0, w_of_labels - w_of_correct_label + 1)

    return L