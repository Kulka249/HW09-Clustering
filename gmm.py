import numpy as np
from sklearn.mixture import GaussianMixture


def concatenate_features(X1, X2):
    n = len(X1)

    X1 = np.reshape(X1, (n, 1))
    X2 = np.reshape(X2, (n, 1))

    X = np.concatenate((X1, X2), axis=1)

    return X


def gaus_mixture(data, n_components):
    gm = GaussianMixture(n_components=n_components[0], random_state=0).fit(data)

    best_bic = gm.bic(data)
    best_no_clusters = n_components[0]

    for k in n_components:
        gm = GaussianMixture(n_components=k, random_state=0).fit(data)
        bic = gm.bic(data)

        if bic < best_bic:
            best_bic = bic
            best_no_clusters = k

    return best_no_clusters



if __name__ == "__main__":
    # load data and reshape to work with GMM implementation
    X1 = np.genfromtxt('gmm_data_x1.csv', delimiter=',')
    X2 = np.genfromtxt('gmm_data_x2.csv', delimiter=',')

    print("Shape of the first feature array: ", X1.shape)
    print("Shape of the second feature array: ", X2.shape)

    # Concatenate individual feature arrays into a single numpy array
    X = concatenate_features(X1,X2)

    print("Shape of the concatenated array: ", X.shape, "\n")

    # call function and get output
    best_k = gaus_mixture(data=X, n_components=[3, 4, 5, 6, 7, 8, 9])
    print('Best fit is when k = %d clusters are used' % (best_k))
