import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt


def MarvellousKMean():
    print("________________________________")
    # Set three centers,the model should predict similar results
    center_1 = np.array([1, 1])
    print(center_1)
    print("________________________________")
    center_2 = np.array([5, 5])
    print(center_2)
    print("________________________________")
    center_3 = np.array([8, 1])
    print(center_3)
    print("________________________________")

    # Generate random data and centre it to three centers
    data_1 = np.random.randn(7, 2)+center_1
    print("Elements of first cluster with size"+str(len(data_1)))
    print(data_1)
    print("________________________________")
    data_2 = np.random.randn(7, 2)+center_2
    print("Elements of first cluster with size"+str(len(data_2)))
    print(data_2)
    print("________________________________")
    data_3 = np.random.randn(7, 2)+center_3
    print("Elements of first cluster with size"+str(len(data_3)))
    print(data_3)
    print("________________________________")
    data = np.concatenate((data_1, data_2, data_3), axis=0)
    print("Size of complete data set"+str(len(data)))
    print(data)
    print("________________________________")
    plt.scatter(data[:, 0], data[:, 1], s=7)
    plt.title('Input Dataset')
    plt.show()
    print("________________________________")

    # Number of clusters
    k = 3

    # Number of training data
    n = data.shape[0]
    print("Total number of elements are", n)
    print("________________________________")
    # Number of features in data
    c = data.shape[1]
    print("Total number of Features are", c)
    print("________________________________")
    # Generate random centers,here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(data, axis=0)
    print("Value of Mean", mean)
    print("________________________________")
    # Calculate standard deviation
    std = np.std(data, axis=0)
    print("Value of std", std)
    print("________________________________")
    centers = np.random.randn(k, c)*std + mean
    print("Random points are", centers)
    print("________________________________")
    # plot the data and centers grenerated random
    plt.scatter(data[:, 0], data[:, 1], c='r', s=7)
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='g', s=150)
    plt.title('Input Data with random centroid *')
    plt.show()
    print("________________________________")

    centers_old = np.zeros(centers.shape)  # to store old centers
    centers_new = deepcopy(centers)  # Store new centers

    print("Values of old centroids")
    print(centers_old)
    print("________________________________")

    print("Values of new centroids")
    print(centers_new)
    print("________________________________")

    data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    print("Initial distances are")
    print(distances)
    print("________________________________")

    error = np.linalg.norm(centers_new - centers_old)
    print("Value of error is", error)
    # When,after an update,the estimate of that center stays the same,exit loop

    while error != 0:
        print("Value of error is", error)
        # Measure the distance to every center
        print("Measure the distance to every center")
        for i in range(k):
            print("Iteration number ", i)
            distances[:, i] = np.linalg.norm(data-centers[i], axis=1)

        # Assign all training data to closest center
        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)

        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new-centers_old)
    # end of while
    centers_new

    # Plot the data and centers grnerated as random
    plt.scatter(data[:, 0], data[:, 1], s=7)
    plt.scatter(centers_new[:, 0], centers_new[:, 1], marker='*', c='g', s=150)
    plt.title('Final data with Centroid')
    plt.show()


def main():
    print("Clustering using K Mean Algorithm")

    MarvellousKMean()


if __name__ == "__main__":
    main()
