import part1_test
import numpy as np


def evaluate(classifications, test_histogram, actual_histogram, print_output=False):
    """Return the confusion matrix for these classifications"""
    classification_accy = np.zeros(10)
    classification_accy_numerator = np.zeros(10)
    classification_accy_denominator = actual_histogram

    #records the number of images from class r classified as class c in test data
    confusion_matrix_numerator = np.zeros((10,10))
    #each row of denominator should be number of images from class r in test data
    confusion_matrix_denominator = np.zeros((10,10))

    for r in range(confusion_matrix_denominator.shape[0]):
        for c in range(confusion_matrix_denominator.shape[1]):
            confusion_matrix_denominator[r][c] = actual_histogram[r]

    for decided_class, actual_class, score, pixels in classifications:
        if decided_class == actual_class:
            classification_accy_numerator[actual_class] += 1

        c = decided_class
        r = actual_class
        confusion_matrix_numerator[r][c] += 1

    classification_accy = classification_accy_numerator / classification_accy_denominator
    confusion = confusion_matrix_numerator / confusion_matrix_denominator

    #find the tokens with the highest/lower posterior probabilities(class scores)
    max_posterior = [-1000000000] * 10
    max_posterior_digit = [None] * 10
    min_posterior = [10000000000] * 10
    min_posterior_digit = [None] * 10

    for decided_class, actual_class, score, pixels in classifications:
        if score > max_posterior[actual_class]:
            max_posterior[actual_class] = score
            max_posterior_digit[actual_class] = pixels
        if score < min_posterior[actual_class]:
            min_posterior[actual_class] = score
            min_posterior_digit[actual_class] = pixels

    if print_output:
        for digit,class_accy in enumerate(classification_accy):
            print("digit {}'s classification accuracy: {}".format(digit, class_accy))
        print()

        print("Confusion Matrix report:")
        for row in confusion:
            print(' '.join('{}   '.format(str(spot))[:4] for spot in row))
        print()

        for digit in range(10):
            print("Digit {}'s maximum posterior digit is:".format(digit))
            for row in max_posterior_digit[digit]:
                print(' '.join(str(int(spot)) for spot in row))
            print()
            print("Digit {}'s minimum posterior digit is:".format(digit))
            for row in min_posterior_digit[digit]:
                print(' '.join(str(int(spot)) for spot in row))
            print()

    return confusion


def main():
    classifications, test_histogram, actual_histogram = part1_test.test()
    return evaluate(classifications, test_histogram, actual_histogram, print_output=True)


if __name__ == '__main__':
    main()
