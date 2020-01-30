from matplotlib import use, cm
use('TkAgg')
import numpy as np
import scipy.io
import scipy.linalg as la
import matplotlib.pyplot as plt
import math

## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easy to
#  visualize.
#
#  Our example case consists of 2 network server statistics across
#  several machines: the latency and throughput of each machine.
#  This exercise will help us find possibly faulty (or very fast) machines.
#
def part1_load_example_dataset(X):
    print("Visualizing example dataset for outlier detection.")


    #  Visualize the example dataset
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.axis([0, 30, 0, 30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()


## ================== Part 2: Estimate the dataset statistics ===================
#  For this exercise, we assume a Gaussian distribution for the dataset.
#
#  We first estimate the parameters of our assumed Gaussian distribution, 
#  then compute the probabilities for each of the points and then visualize 
#  both the overall distribution and where each of the points falls in 
#  terms of that distribution.
#
def part2_estimate_the_dataset_statistics(X):
    print("Visualizing Gaussian fit.")

    #  Estimate mu and sigma2
    mu, sigma2 = estimate_gaussian(X)

    #  Returns the density of the multivariate normal at each data point (row) 
    #  of X
    p = multivariate_gaussian(X, mu, sigma2)
    p2 = multivariate_gaussian2(X, mu, sigma2)

    #  Visualize the fit
    visualize_fit(X,  mu, sigma2)

def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    var = np.mean(np.power(X-mu, 2), axis=0)
    return mu, var

def multivariate_gaussian(X, mu, sigma2):
    k = np.size(mu, 0)
    m, n = np.shape(X)
    sigma2_diag = np.diag(sigma2)
    #print(sigma2_diag)
    #print(np.linalg.det(sigma2_diag))
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(sigma2_diag) ** (-0.5) * \
        np.prod(np.exp(-((X-mu)*(X-mu)).dot(np.diag(1/np.sqrt(2*sigma2)))), axis=1)*(1/np.sqrt(2*math.pi*np.linalg.det(sigma2_diag)))
    #print("=============")
    #print(1/np.sqrt(2*math.pi*np.linalg.det(sigma2_diag)))
    #print(1/(((2*math.pi)*la.det(sigma2_diag))**(0.5)))
    #print(np.shape(1/np.sqrt(2*math.pi*np.linalg.det(sigma2_diag))))
    #print(p)
    return p

def multivariate_gaussian2(x, mu, sigma2):
    k = np.size(mu, 0)
    #print(k)
    sigma2 = np.diag(sigma2)
    x = x-mu
    p = (2*math.pi)**(-k/2)*la.det(sigma2)**(-0.5)*np.exp(-0.5*np.sum(x.dot(la.pinv(sigma2))*x, 1))
    #print("=============")
    #print((2*math.pi)**(-k/2)*la.det(sigma2)**(-0.5))
    #print(np.shape((2*math.pi)**(-k/2)*la.det(sigma2)**(-0.5)))
    #print(p)
    return p

def visualize_fit(X, mu, sigma2):
    temp = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(temp, temp)
    #need to modify multivariate_gaussian with own codes
    z = multivariate_gaussian2(np.vstack((X1.flatten(), X2.flatten())).T, mu, sigma2)
    z = z.reshape(X1.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.contour(X1, X2, z, np.power(10.0, np.arange(-20, 0, 3)))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()


## ================== Part 3: Find Outliers ===================
#  Now you will find a good epsilon threshold using a cross-validation set
#  probabilities given the estimated Gaussian distribution
# 

def part3_find_outliers(X, Xval, yval):
    mu, sigma2 = estimate_gaussian(X)
    pval = multivariate_gaussian2(Xval, mu, sigma2)
    p = multivariate_gaussian2(X, mu, sigma2)
    epsilon, F1 = select_threshold(yval, pval)
    print("Best epsilon found using cross-validation: %e" % epsilon)
    print("Best F1 on Cross Validation Set:  %f" % F1)
    print("   (you should see a value epsilon of about 8.99e-05)")

    #  Find the outliers in the training set and plot the
    outliers = np.where(p < epsilon, True, False)

    #  Draw a red circle around those outliers
    plt.plot(X[outliers, 0], X[outliers, 1], 'ro', lw=2, markersize=10, fillstyle='none', markeredgewidth=1)
    visualize_fit(X,  mu, sigma2)

    plt.show()

def select_threshold(yval, pval):
    """
    finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    best_epsilon = 0
    best_f1 = 0
    stepsize = (np.max(pval) - np.min(pval)) / 1000.0
    for epsilon in np.arange(np.min(pval),np.max(pval), stepsize):
        tp = np.sum(np.logical_and(pval < epsilon, yval == 1))
        fp = np.sum(np.logical_and(pval < epsilon, yval == 0))
        fn = np.sum(np.logical_and(pval >= epsilon, yval == 1))
        if tp+fp == 0 or tp+fn == 0:
            continue
        prec = tp / (tp+fp)
        rec = tp / (tp+fn)
        f1 = 2 * (prec*rec) / (prec+rec)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1

## ================== Part 4: Multidimensional Outliers ===================
#  We will now use the code from the previous part and apply it to a 
#  harder problem in which more features describe each datapoint and only 
#  some features indicate whether a point is an outlier.
#
def part4_multidimensional_outliers():
    #  Loads the second dataset. You should now have the
    #  variables X, Xval, yval in your environment
    data = scipy.io.loadmat('ex8data2.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval'].flatten()

    #  Apply the same steps to the larger dataset
    mu, sigma2 = estimate_gaussian(X)

    #  Training set 
    p = multivariate_gaussian2(X, mu, sigma2)

    #  Cross-validation set
    pval = multivariate_gaussian2(Xval, mu, sigma2)

    #  Find the best threshold
    epsilon, F1 = select_threshold(yval, pval)

    print("Best epsilon found using cross-validation: %e" % epsilon)
    print("Best F1 on Cross Validation Set:  %f" % F1)
    print("# Outliers found: %d" % sum(p < epsilon))
    print("   (you should see a value epsilon of about 1.38e-18)")


def main():
    #  The following command loads the dataset. You should now have the
    #  variables X, Xval, yval in your environment
    data = scipy.io.loadmat('ex8data1.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval'].flatten()

    #part1_load_example_dataset(X)
    #part2_estimate_the_dataset_statistics(X)
    part3_find_outliers(X, Xval, yval)
    part4_multidimensional_outliers()

if __name__ == '__main__':
    main()
