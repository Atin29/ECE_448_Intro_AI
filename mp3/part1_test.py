from math import log
import numpy as np
import part1_train


def test(print_output=False):
    """Return a tuple of (list, numpy array, numpy array):
        1. list of tuples of (our classification, actual classification, classification score)
        2. shape (10,) numpy array - histogram of our classifications
        3. shape (10,) numpy array - histogram of actual classifications
    """
    part1_train.train()

    classifications = []
    test_classifications_hist = np.zeros(10)
    actual_classifications_hist = np.zeros(10)

    with open("digitdata/optdigits-orig_test.txt", 'r') as f:
        digit_index_y = 0
        digit_pixels = np.zeros((32,32))

        for line_num, line in enumerate(f, start=1):
            if line_num % 33 == 0:
                if print_output:
                    for row in digit_pixels:
                        print(' '.join(str(int(spot)) for spot in row))
                # get the actual classification from this line
                actual_classification = int(line[1])
                if print_output:
                    print(actual_classification)
                actual_classifications_hist[actual_classification] += 1

                # run our own classification
                best_classification = 0
                best_classification_score = None
                for potential_class in range(10):
                    class_score = log(part1_train.priors[potential_class])
                    class_likelihoods_1 = part1_train.likelihoods_1[potential_class]
                    class_likelihoods_0 = part1_train.likelihoods_0[potential_class]
                    for row_idx, row in enumerate(digit_pixels):
                        for col_idx, pixel in enumerate(row):
                            if pixel == 0:
                                class_score += log(class_likelihoods_0[row_idx][col_idx])
                            else:
                                class_score += log(class_likelihoods_1[row_idx][col_idx])
                    if print_output:
                        print('score for potential class {}: {}'.format(potential_class, class_score))
                    if best_classification_score is None or class_score > best_classification_score:
                        best_classification = potential_class
                        best_classification_score = class_score
                if print_output:
                    print('best classification: {} with score {}'.format(best_classification, best_classification_score))

                test_classifications_hist[best_classification] += 1

                classifications.append((best_classification, actual_classification, best_classification_score, digit_pixels))

                digit_index_y = 0
                digit_pixels = np.zeros((32,32))
                if print_output:
                    print()
            else:
                # build up this digit's pixel matrices
                for digit_index_x, ch in enumerate(line[:-1]):
                    digit_pixels[digit_index_y][digit_index_x] = int(ch)
                digit_index_y += 1

    if print_output:
        print('test classes histogram:   {}'.format(test_classifications_hist))
        print('actual classes histogram: {}'.format(actual_classifications_hist))
    return classifications, test_classifications_hist, actual_classifications_hist


def main():
    test(print_output=True)


if __name__ == '__main__':
    main()
