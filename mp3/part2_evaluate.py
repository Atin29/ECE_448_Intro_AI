import numpy as np
import part2_test


def evaluate(classifications, test_histogram, actual_histogram):
    confusion = np.zeros((10,10))
    # records the number of images from class r classified as class c in test data
    confusion_matrix_numerator = np.zeros((10,10))
    # each row of denominator should be number of images from class r in test data
    confusion_matrix_denominator = np.zeros((10,10))

    for r in range(10):
        for c in range(10):
            confusion_matrix_denominator[r][c] = actual_histogram[r]

    # the number of successful classifications for each digit
    num_successful_classifications = np.zeros(10)

    for decided_class, actual_class, dot_prod in classifications:
        if decided_class == actual_class:
            num_successful_classifications[actual_class] += 1

        confusion_matrix_numerator[actual_class][decided_class] += 1

    classification_accuracies = num_successful_classifications / actual_histogram
    
    confusion = confusion_matrix_numerator / confusion_matrix_denominator
    print('confusion matrix:')
    for row in confusion:
        print(' '.join('{}   '.format(str(spot))[:4] for spot in row))
    print()
    print('successful classifications per class: {}'.format(num_successful_classifications))
    print('individual class accuracies: {}'.format(classification_accuracies))
    print('overall accuracy: {}'.format(sum(num_successful_classifications) / sum(actual_histogram)))


def main():
    classifications, test_histogram, actual_histogram = part2_test.test()
    evaluate(classifications, test_histogram, actual_histogram)

if __name__ == '__main__':
    main()
