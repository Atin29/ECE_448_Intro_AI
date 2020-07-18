import numpy as np

LAPLACE_SMOOTHING_CONSTANT = 0.1

#array that contains the number of times each class occurs in training data
class_counts = np.zeros(10)

#array that contains the priors of each class
priors = np.zeros(10)

#lists that contains 2D arrays of pixel counts for each class
pixel_counts_1 = []
pixel_counts_0 = []

#lists that contains 2D arrays of likelihoods of each pixel for each class
likelihoods_1 = []
likelihoods_0 = []

for _ in range(10):
    pixel_counts_1.append(np.zeros((32,32)))
    pixel_counts_0.append(np.zeros((32,32)))
    likelihoods_1.append(np.zeros((32,32)))
    likelihoods_0.append(np.zeros((32,32)))


def train(print_output=False):
    with open("digitdata/optdigits-orig_train.txt", 'r') as f:
        digit_index_x = 0
        digit_index_y = 0
        temp_pixels_0 = np.zeros((32,32))
        temp_pixels_1 = np.zeros((32,32))
        for line_num, line in enumerate(f, start=1):
            #line detailing class of figure
            if line_num % 33 == 0:
                digit = int(line[1])
                class_counts[digit] += 1
                pixel_counts_1[digit] += temp_pixels_1
                pixel_counts_0[digit] += temp_pixels_0
                digit_index_x = 0
                digit_index_y = 0
                temp_pixels_0 = np.zeros((32,32))
                temp_pixels_1 = np.zeros((32,32))
            else:
                for ch in line:
                    if ch == '1':
                        temp_pixels_1[digit_index_y][digit_index_x] += 1
                    if ch == '0':
                        temp_pixels_0[digit_index_y][digit_index_x] += 1
                    digit_index_x += 1
                digit_index_x = 0
                digit_index_y += 1

    global priors
    priors = class_counts / sum(class_counts)

    for digit, heatmap in enumerate(pixel_counts_1):
        likelihoods_1[digit] = (heatmap + LAPLACE_SMOOTHING_CONSTANT) \
                                / (class_counts[digit] + LAPLACE_SMOOTHING_CONSTANT*2)

    for digit, heatmap in enumerate(pixel_counts_0):
        likelihoods_0[digit] = (heatmap + LAPLACE_SMOOTHING_CONSTANT) \
                                / (class_counts[digit] + LAPLACE_SMOOTHING_CONSTANT*2)

    if print_output:
        for digit, heatmap in enumerate(pixel_counts_1):
            print("digit {} 1's heatmap:".format(digit))
            for row in heatmap:
                row_string = []
                for spot in row:
                    spot_string = '{0:3}'.format(str(int(spot)))
                    row_string.append(spot_string)
                print(' '.join(row_string))
            print()

        for digit, heatmap in enumerate(pixel_counts_0):
            print("digit {} 0's heatmap:".format(digit))
            for row in heatmap:
                row_string = []
                for spot in row:
                    spot_string = '{:3}'.format(str(int(spot)))
                    row_string.append(spot_string)
                print(' '.join(row_string))
            print()

        for digit, frequency in enumerate(class_counts):
            print('digit {} occurs {} times in the training data' \
                    .format(digit, int(frequency)))

        for digit, prior in enumerate(priors):
            print('prior for digit {} is {}'.format(digit, prior))

        for digit, heatmap in enumerate(likelihoods_1):
            print("digit {} 1's likelihood:".format(digit))
            for row in heatmap:
                print(' '.join('{}   '.format(str(spot))[:4] for spot in row))
            print()

        for digit, heatmap in enumerate(likelihoods_0):
            print("digit {} 0's likelihood:".format(digit))
            for row in heatmap:
                print(' '.join('{}   '.format(str(spot))[:4] for spot in row))
            print()


def main():
    train(print_output=True)

if __name__ == '__main__':
    main()
