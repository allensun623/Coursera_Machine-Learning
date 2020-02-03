## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm 
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you shoudl complete the code in the findClosestCentroids function. 
#
from matplotlib import use, cm
use('TkAgg')
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import random
import pysnooper


def part1_find_closest_centroids():
    print("Finding closest centroids.")

    # Load an example dataset that we will be using
    data = scipy.io.loadmat('ex7data2.mat')
    X = data['X']

    # Select an initial set of centroids
    K = 3 # 3 Centroids
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    # Find the closest centroids for the examples using the
    # initial_centroids
    idx = find_closest_centroids(X, initial_centroids)

    print("Closest centroids for the first 3 examples:")
    print(idx[0:3].tolist())
    print("(the closest centroids should be 0, 2, 1 respectively)")
    return X, idx, K

def find_closest_centroids(X, centroids):
    # Set K
    K = len(centroids)
    idx = np.zeros(X.shape[0])
    for i in range(len(idx)):
        distance = [np.sum((X[i] - k_x)**2) for k_x in centroids]
        idx[i] = distance.index(min(distance))   

    return idx

## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
def part2_compute_means():
    X, idx, K = part1_find_closest_centroids()
    print("Computing centroids means.")

    #  Compute means based on the closest centroids found in the previous part.
    centroids = compute_centroids(X, idx, K)

    print("Centroids computed after initial finding of closest centroids:")
    for c in centroids:
        print(c)

    print("(the centroids should be")
    print("   [ 2.428301 3.157924 ]")
    print("   [ 5.813503 2.633656 ]")
    print("   [ 7.119387 3.616684 ]")

def compute_centroids(X, idx, K):
    centroids = [np.mean(X[np.where(idx==i)], axis=0) for i in range(K)]
    return np.array(centroids)

## =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided. 
#
def part3_kMeans_clustering():
    print("Running K-Means clustering on example dataset.")

    # Load an example dataset
    data = scipy.io.loadmat('ex7data2.mat')
    X = data['X']
    X_mu = np.mean(X)
    X_sigma = np.std(X, ddof=1)
    #X = (X-X_mu) / X_sigma

    # Settings for running K-Means
    K = 3
    max_iters = 20

    # For consistency, here we set centroids to specific values
    # but in practice you want to generate them automatically, such as by
    # settings them to be random examples (as can be seen in
    # kMeansInitCentroids).
    initial_centroids = kMeans_init_centroids(X, K)

    # Run K-Means algorithm. The 'true' at the end tells our function to plot
    # the progress of K-Means
    centroids, idx = run_kMeans(X, initial_centroids, max_iters, K, True)
    print("K-Means Done.")

def run_kMeans(X, centroids, max_iters, K, plot_progress=False):
    idx = find_closest_centroids(X, centroids)
    for i in range(max_iters):
        new_centroids = compute_centroids(X, idx, K)
        #centroids movement trail
        if plot_progress:
            for c in range(K):
                plot_x = np.column_stack((centroids[c:c+1,:1], new_centroids[c:c+1,:1]))
                plot_y = np.column_stack((centroids[c:c+1,1:], new_centroids[c:c+1,1:]))
                plt.plot(plot_x[0], plot_y[0], 'kx-')
        centroids = new_centroids
        idx = find_closest_centroids(X, centroids)
    if plot_progress:
        #plot clusters
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'w', 'k']
        for k in range(K):
            c_k = np.where(idx==k)
            p_k = plt.scatter(X[c_k][:,:1], 
                        X[c_k][:,1:], 
                        marker='o', 
                        facecolors='none', 
                        edgecolors=color_list[k])
        plt.scatter(centroids[:,:1], centroids[:,1:], marker='x', color='black')    
        plt.title('Iteration number %d' % max_iters)
        plt.show()
    return centroids, idx


        

## ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel on to it's closest centroid.
#  
#  You should now complete the code in kMeansInitCentroids.m
#

def part4_kMeans_clustering_on_pixels():
    print("Running K-Means clustering on pixels from an image.")

    #  Load an image of a bird
    A = scipy.misc.imread('rainbow_copy.jpg')

    # If imread does not work for you, you can try instead
    #   load ('bird_small.mat')

    A = A / 255.0 # Divide by 255 so that all values are in the range 0 - 1

    # Size of the image
    img_size = A.shape

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = A.reshape(img_size[0] * img_size[1], 3)

    # Run your K-Means algorithm on this data
    # You should try different values of K and max_iters here
    K = 7
    max_iters = 10

    # When using K-Means, it is important to initialize the centroids
    # randomly. 
    # You should complete the code in kMeansInitCentroids.m before proceeding
    initial_centroids = kMeans_init_centroids(X, K)

    # Run K-Means
    centroids, idx = run_kMeans(X, initial_centroids, max_iters, K, False)
    return X, idx, centroids, img_size, A, K

def kMeans_init_centroids(X, K):
    initial_centroids = random.sample(X.tolist(),  K)
    return np.array(initial_centroids)

## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we 

def part5_image_compression():
    print("Applying K-Means to compress an image.")
    X, idx, centroids, img_size, A, K = part4_kMeans_clustering_on_pixels()
    # Essentially, now we have represented the image X as in terms of the
    # indices in idx. 

    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by it's index in idx) to the centroid value
    X_recovered = np.array([centroids[int(e)] for e in idx])

    # Reshape the recovered image into proper dimensions
    X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

    # Display the original image 
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title('Original image size: %i*%i' % (img_size[0], img_size[1]))

    # Display compressed image side by side
    plt.subplot(1, 2, 2)
    plt.imshow(X_recovered)
    plt.title('Compressed, with %d colors.' % K)
    plt.show()


def main():
    #part1_find_closest_centroids()    
    #part2_compute_means()
    part3_kMeans_clustering()
    #part4_kMeans_clustering_on_pixels()
    #part5_image_compression()

if __name__ == "__main__":
    main()