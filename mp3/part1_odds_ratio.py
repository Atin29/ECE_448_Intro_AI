import part1_test
import part1_train
import part1_evaluation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from part1_train import likelihoods_1, likelihoods_0


def visualize_odds_ratio(confusion):

    most_confused_values = []
    #find the 4 largest numbers not in diagonal for confusion matrix:
    for row_idx, row in enumerate(confusion):
        for col_idx, cell in enumerate(row):
            if row_idx != col_idx and cell != 0:
                most_confused_values.append(cell)
    most_confused_values.sort(reverse=True)
    most_confused_values = most_confused_values[0:4]

    print("Most confused values are:")
    print(most_confused_values)

    #list that stores the most confused pairs of digits as tuples (digitA, digitB)
    #digitA is accidentally classified into digitB
    most_confused_pairs = []
    #used most_confused_values to find the 4 most confused pairs of digits
    break_loop = False
    for num in most_confused_values:
        for row_idx, row in enumerate(confusion):
            for col_idx, cell in enumerate(row):
                if cell == num:
                    most_confused_pairs.append((row_idx, col_idx))
                if len(most_confused_pairs) == 4:
                    break_loop = True
                    break
            if break_loop == True:
                break
        if break_loop == True:
            break


    print("Most confused pairs are:")
    print(most_confused_pairs)

    #plot likelihood of digitA, likelihood of digitB, odds ratio of digitA & digitB

    #convert likelihoods into log likelihoods
    log_likelihoods_1 = np.log(likelihoods_1)
    log_likelihoods_0 = np.log(likelihoods_0)

    subplot_num = 1
    for confused_pair in most_confused_pairs:
        digitA = confused_pair[0]
        digitB = confused_pair[1]

        #plot '1' likelihood heatmap of digitA
        pixelsA = log_likelihoods_1[digitA]
        plt.subplot(4,3,subplot_num)
        plot_likelihoods_A = sns.heatmap(pixelsA, cmap='jet', vmin=-4, vmax=0)
        subplot_num += 1

        #plot '1' likelihood heatmap of digitB
        pixelsB = log_likelihoods_1[digitB]
        plt.subplot(4,3,subplot_num)
        plot_likelihoods_A = sns.heatmap(pixelsB, cmap='jet', vmin=-4, vmax=0)
        subplot_num += 1

        #plot odds ratio of the the two digits
        plt.subplot(4,3,subplot_num)
        odds_ratio = log_likelihoods_1[digitA] - log_likelihoods_1[digitB]
        plot_odds_ratio = sns.heatmap(odds_ratio, cmap='jet', vmin=-3, vmax=2)
        subplot_num += 1

    plt.show()


def main():
    # classifications, test_histogram, actual_histogram = part1_test.test()
    # part1_train.train()
    confusion = part1_evaluation.main()
    visualize_odds_ratio(confusion)

if __name__ == '__main__':
    main()