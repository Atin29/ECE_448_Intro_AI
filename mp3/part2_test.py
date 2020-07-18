import numpy as np
import part2_train


def test(print_output=False):
    """Return a tuple of (list, numpy array, numpy array):
        1. list of tuples of (our classification, actual classification, classification score)
        2. shape (10,) numpy array - histogram of our classifications
        3. shape (10,) numpy array - histogram of actual classifications
    """
    weights = part2_train.train()

    classifications = []
    test_classifications_hist = np.zeros(10)
    actual_classifications_hist = np.zeros(10)

    with open("digitdata/optdigits-orig_test.txt", 'r') as f:
        pixels = np.zeros(1024)
        pixels_index = 0
        misclassified = 0

        for line_num, line in enumerate(f, start=1):
                if line_num % 33 == 0:
                    actual_class = int(line[1])
                    actual_classifications_hist[actual_class] += 1
                    pixels_index = 0

                    # class = argmax_c <w_c, x>
                    dot_prod, decided_class = max((np.dot(weights[c], pixels), c) for c in range(10))
                    test_classifications_hist[decided_class] += 1

                    classifications.append((decided_class, actual_class, dot_prod))

                    if decided_class != actual_class:
                        misclassified += 1

                    pixels = np.zeros(1024)
                else:
                    for ch in line[:-1]:
                        pixels[pixels_index] = int(ch)
                        pixels_index += 1

    if print_output:
        print('test classes histogram:   {}'.format(test_classifications_hist))
        print('actual classes histogram: {}'.format(actual_classifications_hist))
    return classifications, test_classifications_hist, actual_classifications_hist


def main():
    test(print_output=True)


if __name__ == '__main__':
    main()
