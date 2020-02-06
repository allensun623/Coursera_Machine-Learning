## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#

from matplotlib import use
use('TkAgg')
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.colors as pltcolor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
from displayData import displayData
import random
import matplotlib.cm as cm
from ex7 import find_closest_centroids
from ex7 import compute_centroids
from ex7 import run_kMeans
from ex7 import kMeans_init_centroids

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize

def part1_load_example_dataset():
    print("Visualizing example dataset for PCA.")
    #  The following command loads the dataset. You should now have the 
    #  variable X in your environment
    data = scipy.io.loadmat('ex7data1.mat')
    X = data['X']

    #  Visualize the example dataset
    plt.scatter(X[:, 0], X[:, 1], marker='o', color='b', facecolors='none', lw=1.0)
    plt.axis([0.5, 6.5, 2, 8])
    plt.axis('equal')
    plt.show()


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
def part2_principal_component_analysis():
    print("Running PCA on example dataset.")
    data = scipy.io.loadmat('ex7data1.mat')
    X = data['X']

    #  Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = feature_normalize(X)

    #  Run PCA
    U, S, V = pca(X_norm)

    #  Compute mu, the mean of the each feature

    #  Draw the eigenvectors centered at mean of data. These lines show the
    #  directions of maximum variations in the dataset.
    mu2 = mu + 1.5*np.stack((S, S), axis=1)*U
    plt.plot([mu[0], mu2[0, 0]], [mu[1], mu2[0, 1]], '-k', lw=2)
    plt.plot([mu[0], mu2[1, 0]], [mu[1], mu2[1, 1]], '-k', lw=2)
    #  Visualize the example dataset
    plt.scatter(X[:, 0], X[:, 1], marker='o', color='b', facecolors='none', lw=1.0)
    plt.axis([0.5, 6.5, 2, 8])
    plt.axis('equal')
    plt.show()

    print("Top eigenvector: ")
    print(" U(:,1) = %f %f ", U[0,0], U[1,0])
    print("(you should expect to see -0.707107 -0.707107)")

def feature_normalize(X):
    X_mu = np.mean(X, axis=0)
    X_sigma = np.std(X, ddof=1, axis=0)
    X_norm = (X-X_mu) / X_sigma
    return X_norm, X_mu, X_sigma 

def pca(X):
    m, n = np.shape(X) 
    sigma = 1/m*X.T.dot(X)
    U, S, V = la.svd(sigma)
    return U, S, V

## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
def part3_dimension_reduction():
    print("Dimension reduction on example dataset.")
    data = scipy.io.loadmat('ex7data1.mat')
    X = data['X']

    #  Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = feature_normalize(X)

    #  Run PCA
    U, S, V = pca(X_norm)
    #  Plot the normalized dataset (returned from pca)
    plt.figure()
    plt.scatter(X_norm[:, 0], X_norm[:, 1], marker='o', color='b', facecolors='none', lw=1.0)
    plt.axis([-4, 3, -4, 3]) #axis square
    plt.axis('equal')
    plt.show()

    #  Project the data onto K = 1 dimension
    K = 1
    Z = project_data(X_norm, U, K)
    print("Projection of the first example: %f", Z[0])
    print("(this value should be about 1.481274)")

    X_rec  = recover_data(Z, U, K)
    print("Approximation of the first example: %f %f"% (X_rec[0, 0], X_rec[0, 1]))
    print("(this value should be about  -1.047419 -1.047419)")

    #  Draw lines connecting the projected points to the original points
    plt.scatter(X_rec[:, 0], X_rec[:, 1], marker='o', color='r', facecolor='none', lw=1.0)
    for i in range(len(X_norm)):
        plt.plot([X_norm[i, 0], X_rec[i, 0]], [X_norm[i, 1], X_rec[i, 1]], '--k')

    plt.show()
    k_value = k_value_compute(X_norm, X_rec)




def project_data(X_norm, U, K):
    Z = X_norm.dot(U[:, 0:K])
    return Z

def recover_data(Z, U, K):
    print(np.shape(Z))
    print(np.shape(U))
    print(np.shape(U[:, 0:K]))
    X_rec = Z.dot(U[:, 0:K].T)
    return X_rec

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
def part4_loading_and_visualizing_face_data():
    print("Loading face dataset.")

    #  Load Face dataset
    data = scipy.io.loadmat('ex7faces.mat')
    X = data['X']

    #  Display the first 100 faces in the dataset
    displayData(X[0:100, :])
    return X
## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#

def part5_PCA_on_face_data_eigenfaces():
    X = part4_loading_and_visualizing_face_data()
    print("Running PCA on face dataset.\n(this might take a minute or two ...)\n\n")

    #  Before running PCA, it is important to first normalize X by subtracting 
    #  the mean value from each feature
    X_norm, mu, sigma = feature_normalize(X)

    #  Run PCA
    U, S, V = pca(X_norm)

    #  Visualize the top 36 eigenvectors found
    displayData(U[:, 1:36].T)
    return X_norm, U

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
def part6_dimension_reduction_for_faces():
    X_norm, U = part5_PCA_on_face_data_eigenfaces()
    print("Dimension reduction for face dataset.")

    K = 300
    Z = project_data(X_norm, U, K)

    print("The projected data Z has a size of: ")
    print("%d %d"% Z.shape)
    return K, X_norm, Z, U

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

def part7_visualization_of_faces_after_PCA_dimension_reduction():
    print("Visualizing the projected (reduced dimension) faces.")
    K, X_norm, Z, U = part6_dimension_reduction_for_faces()
    X_rec  = recover_data(Z, U, K)
    # Display normalized data
    plt.subplot(1, 2, 1)
    displayData(X_norm[:K,:])
    plt.title('Original faces')
    plt.axis('equal')

    # Display reconstructed data from only k eigenfaces
    plt.subplot(1, 2, 2)
    displayData(X_rec[:K,:])
    plt.title('Recovered faces')
    plt.axis('equal')
    plt.show()
    k_value = k_value_compute(X_norm, X_rec)


## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.
def part_8_a():
    # Re-load the image from the previous exercise and run K-Means on it
    # For this to work, you need to complete the K-Means assignment first
    A = scipy.misc.imread('bird_small.png')

    # If imread does not work for you, you can try instead
    #   load ('bird_small.mat')

    A = A / 255.0
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], 3)
    K = 20 
    max_iters = 10
    initial_centroids = kMeans_init_centroids(X, K)
    centroids, idx = run_kMeans(X, initial_centroids, max_iters, K, False)

    #  Sample 1000 random indexes (since working with all the data is
    #  too expensive. If you have a fast computer, you may increase this.
    sel = np.floor(np.random.random(1000) * len(X)) + 1

    #  Setup Color Palette

    #  Visualize the data and centroid memberships in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Xs = np.array([X[int(s)] for s in sel])
    xs = Xs[:, 0]
    ys = Xs[:, 1]
    zs = Xs[:, 2]
    cmap = plt.get_cmap("jet")
    idxn = sel.astype('float')/max(sel.astype('float'))
    colors = cmap(idxn)
    # ax = Axes3D(fig)
    ax.scatter3D(xs, ys, zs=zs, edgecolors=colors, marker='o', facecolors='none', lw=0.4, s=10)

    plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    plt.show()
    return X, K, sel, idx



## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization
def part_8_b():
    X, K, sel, idx = part_8_a()
    # Subtract the mean to use PCA
    X_norm, mu, sigma = feature_normalize(X)

    # PCA and project the data to 2D
    U, S, V = pca(X_norm)
    Z = project_data(X_norm, U, 2)

    X_rec  = recover_data(Z, U, 2)
    # Plot in 2D
    plt.figure()
    zs = np.array([Z[int(s)] for s in sel])
    idxs = np.array([idx[int(s)] for s in sel])
    vec_sel = sel.astype(int)
    colors = cm.rainbow(np.linspace(0, 1, K))
    plt.scatter(Z[vec_sel, 0], Z[vec_sel, 1], c=idx[vec_sel], cmap=pltcolor.ListedColormap(colors), marker='o')
    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    plt.show()
    
    map = plt.get_cmap("jet")
    idxn = idx.astype('float')/max(idx.astype('float'))
    colors = map(idxn)
    plt.scatter(Z[:, 0], Z[:, 1], edgecolors=colors, marker='o', facecolors='none', lw=0.5)
    plt.show()

def k_value_compute(X_norm, X_rec):
    m, _ = np.shape(X_norm)
    k_numerator = 1 / m * np.sum((X_norm-X_rec)*(X_norm-X_rec))
    k_denominator = 1 / m * np.sum(X_norm*X_norm)    
    k_value = k_numerator / k_denominator
    print(k_value)
    return k_value

def main():
    #part1_load_example_dataset()
    part2_principal_component_analysis()
    part3_dimension_reduction()
    part5_PCA_on_face_data_eigenfaces()
    part7_visualization_of_faces_after_PCA_dimension_reduction()
    #part_8_b()

if __name__ == '__main__':
    main()












