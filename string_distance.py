from nltk.metrics import binary_distance

"""
Hamming distance is a metric for comparing two binary data strings.
While comparing two binary strings of equal length,
Hamming distance is the number of bit positions in which the two bits are different.

In this solution strings can be different length
"""
def hamming_strings_distance(str1, str2):
    biggest_string = str1
    smaller_string = str2
    if len(str1) < len(str2):
        biggest_string = str2
        smaller_string = str1

    diff_count = 0

    for i in range(len(smaller_string)):
        if smaller_string[i] != biggest_string[i]:
            diff_count += 1

    return diff_count + len(biggest_string) - len(smaller_string)


if __name__ == "__main__":
    print(binary_distance("hilko", "hello"))