import numpy as np

INITIAL_LEARNING_RATE = 0.2

def initial_weights():
    """Return a list of initial weight vectors for the digit classes"""
    weights = []
    for _ in range(10):
        weights.append(np.random.rand(1024))
    return weights

def learning_rate_decay_function(epoch, learning_rate):
    """Return the new learning rate as a function of the epoch and
    current learning rate."""
    return learning_rate / 1.1

def train(num_epochs=None, print_output=False):
    #array that contains the number of times each class occurs in training data
    class_counts = np.zeros(10)

    #list of weight vectors for the digit classes
    weights = initial_weights()

    epoch_training_results = {}
    epoch = 0

    learning_rate = INITIAL_LEARNING_RATE

    with open("digitdata/optdigits-orig_train.txt", 'r') as f:
        lines = f.readlines()

    prev_weights = None
    while (num_epochs is None or epoch < num_epochs) and (prev_weights is None or \
            not all(np.array_equal(prev_weights[i], weights[i]) for i in range(10))):
        prev_weights = [weights[i] for i in range(10)]
        epoch += 1
        pixels = np.zeros(1024)
        pixels_index = 0
        misclassified = 0
        for line_num, line in enumerate(lines, start=1):
            if line_num % 33 == 0:
                actual_class = int(line[1])
                if epoch == 1:
                    class_counts[actual_class] += 1
                pixels_index = 0

                # class = argmax_c <w_c, x>
                dot_prod, decided_class = max((np.dot(weights[c], pixels), c) for c in range(10))

                if decided_class != actual_class:
                    misclassified += 1
                    # Actual class c, decided class c'
                    # Update for c : w_c  = w_c  + ηx
                    weights[actual_class] = weights[actual_class] + learning_rate*pixels

                    # Update for c': w_c' = w_c' – ηx
                    weights[decided_class] = weights[decided_class] - learning_rate*pixels

                #print(actual_class, decided_class, pixels)
                pixels = np.zeros(1024)
            else:
                for ch in line[:-1]:
                    pixels[pixels_index] = int(ch)
                    pixels_index += 1

        if print_output:
            num_examples = int(sum(class_counts))
            epoch_accuracy = (num_examples - misclassified) / num_examples
            print('training epoch {} accuracy: {}'.format(epoch, str(epoch_accuracy)[:5]))
            print('training at epoch {} misclassified {} training examples'.format(epoch, misclassified))
        epoch_training_results[epoch] = misclassified
        learning_rate = learning_rate_decay_function(epoch, learning_rate)

    if print_output:
        for digit, frequency in enumerate(class_counts):
            print('digit {} occurs {} times in the training data'
                    .format(digit, int(frequency)))

    return weights

def main():
    train(print_output=True)

if __name__ == '__main__':
    main()
