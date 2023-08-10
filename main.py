# Importing necessary packages
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# Creating KNN class
class KnnClassifier(object):
    def __init__(self, k, num_samples, num_features):
        self.k = k
        self._x_train = np.zeros((num_samples, num_features))
        self._y_train = np.zeros((num_samples,))

    def fit(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train

    # finding distance between train data and test data
    def _euclidean_distance(self, x_test):
        euc_distance = np.sqrt(np.sum(np.power((self._x_train - x_test), 2)), axis=1)
        return euc_distance

    # sorting features from low to higher,and append index of them in a list
    def _sort(self, euc_distance):
        list_of_indices = np.argsort(euc_distance)[:, :self.k]
        return list_of_indices

    # after finding the smallest distance we select lowest distance ,k_numbers
    def _count_labels(self, list_of_indices):

    # Finally,here we check which class label is the nearest data to our train_data
    def predict(self, x_test):
        euc_distance = self._euclidean_distance(x_test)
        list_of_indices = self._sort(euc_distance)
        unique_labels, counts = np.unique(list_of_indices, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        print(most_common_label)


def main():
    x_train = np.random.randn(100, 2)
    y_train = np.random.randint(2, size=100)
    x_test = 2 + np.random.randn(100, 2)
    knn = KnnClassifier(k=10, num_samples=100, num_features=2)
    knn.fit(x_train, y_train)
    knn.predict(x_train)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

    """
    # finding best value of k
    def cross_validate(x_train, k_values, num_folds):
        best_k = None
        best_accuracies_per_k = np.zeros((num_folds, len(k_values)))
        accuracies = np.zeros(num_folds)
        fold_size = len(x_train) / num_folds
        for k_idx, k in tqdm(k_values):
            for fold in num_folds:
                start_idx = int(fold * fold_size)
                end_idx = int((fold + 1) * fold_size)
                x_valid_fold = x_train[start_idx:end_idx]
                x_train_fold = np.concatenate([x_train[:start_idx], x_train[end_idx:]], axis=0)
                result = KnnClassifier.nearest_neighbour(x_train_fold, x_valid_fold)
                accuracies[fold] = np.append(f1_score(x_train_fold, result, average='weighted'))
            best_accuracies_per_k.np.append(accuracies)
        best_accuracy_average = np.mean(best_accuracies_per_k, axis=0)
        best_k = np.argmax(best_accuracy_average)
        best_accuracy = np.max(best_accuracy_average)
        return best_k, best_accuracy


    # providing data
    num_samples_per_class = 100
    x_train = np.random.randn(num_samples_per_class, 2)
    x_test = 5 + np.random.randn(num_samples_per_class, 2)

    for i in tqdm(range(100)):
        train_list = []
        output = KnnClassifier.nearest_neighbour(x_train, x_test)
        train_list.append(output)


    # plotting
    def plot(x_train, x_test):
        plt.plot(x_train, "r")
        plt.plot(x_test, "b")
        plt.xlabel("sample")
        plt.ylabel("label")
        plt.legend(["Train Data", "Test Data"])
        plt.title("KNN Classifier")
        plt.show()
"""
