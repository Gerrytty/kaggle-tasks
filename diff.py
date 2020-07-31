from matplotlib import pyplot as plt


def diff(x, f):
    e = 0.001
    return abs((f(x + e) - f(x)) / e)


def diff2(x, f):
    e = 0.001

    return f(x - 2 * e) - 8 * f(x - e) + 8 * f(x + e) - f(x + 2 * e) / 12 * e


def func(x):
    return x * x


if __name__ == "__main__":
    x = [-i for i in range(1000)] + [i for i in range(1000)]
    y = [func(i) for i in x]

    arr = []
    for i in range(len(x)):
        arr.append((diff2(x[i], func), x[i]))

    # point of minimum
    print(min(arr))

    plt.scatter(x, y)
    plt.show()
