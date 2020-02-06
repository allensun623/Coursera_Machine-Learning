## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
from matplotlib import use, cm
use('TkAgg')
import numpy as np
import scipy.io
from sklearn import svm
import matplotlib.pyplot as plt

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#
def part1_loading_and_visualizing_data():
    print("Loading and Visualizing Data ...")

    # Load from ex6data1: 
    # You will have X, y in your environment
    data = scipy.io.loadmat('ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()

    # Plot training data
    plot_data(X, y)
    plt.show()

def plot_data(X, y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    p1 = plt.plot(X[pos, 0], X[pos, 1], 'k+', lw=1, ms=7)
    p2 = plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=7)



## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#
def part2_training_linear_SVM():
    # Load from ex6data1:
    # You will have X, y in your environment
    data = scipy.io.loadmat('ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()

    print("Training Linear SVM ...")

    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)

    C = 100.0
    clf = svm.SVC(C=C, kernel='linear')
    model = clf.fit(X, y)
    visualize_boundary_linear(X, y, model)
    plt.show()

def visualize_boundary_linear(X, y, model):
    """plots a linear decision boundary
    learned by the SVM and overlays the data on it
    """
    w = model.coef_.flatten()
    b = model.intercept_.flatten()
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0]*xp + b)/w[1]
    plt.plot(xp, yp, '-b')
    plot_data(X, y)



## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
def part3_implementing_Gaussian_Kernel():
    print("Evaluating the Gaussian Kernel ...")

    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussian_kernel(x1, x2, sigma)
    
    print("Gaussian Kernel between x1 = [1 2 1], x2 = [0 4 -1], sigma = %0.5f, sim = %f \
            \n(this value should be about 0.324652)\n" % (sigma, sim))

def gaussian_kernel(x1, x2, sigma):
    sim = np.exp(-(x1-x2).T.dot(x1-x2)/(2*sigma**2))
    return sim

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#
def part4_visualizing_dataset2():
    print("Loading and Visualizing Data ...")

    # Load from ex6data2:
    # You will have X, y in your environment
    data = scipy.io.loadmat('ex6data2.mat')
    X = data['X']
    y = data['y'].flatten()

    # Plot training data
    plot_data(X, y)
    plt.show()

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the
#  SVM classifier.
#
def part5_training_SVM_with_RBF_Kernel_dataset2():
    
    print("Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...")

    # Load from ex6data2:
    # You will have X, y in your environment
    data = scipy.io.loadmat('ex6data2.mat')
    X = data['X']
    y = data['y'].flatten()

    # SVM Parameters
    C = 1
    sigma = 0.1
    gamma = 1.0 / (2.0 * sigma ** 2)

    # We set the tolerance and max_passes lower here so that the code will run
    # faster. However, in practice, you will want to run the training to
    # convergence.

    clf = svm.SVC(C=C, kernel='rbf', tol=1e-3, gamma=gamma)
    model = clf.fit(X, y)
    visualize_boundary(X, y, model)
    plt.show()

def visualize_boundary(X, y, model):
    """plots a non-linear decision boundary learned by the
    SVM and overlays the data on it"""

    # Plot the training data on top of the boundary
    plot_data(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:,0]), max(X[:,0]), X.shape[0]).T
    x2plot = np.linspace(min(X[:,1]), max(X[:,1]), X.shape[0]).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)

    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(this_X)

    # Plot the SVM boundary
    #contour(X1, X2, vals, [0 0], 'Color', 'b')
    plt.contour(X1, X2, vals, levels=[0.0, 0.0])

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#
def part6_visualizing_dataset3():
    print("Loading and Visualizing Data ...")

    # Load from ex6data3:
    # You will have X, y in your environment
    data = scipy.io.loadmat('ex6data3.mat')
    X = data['X']
    y = data['y'].flatten()

    # Plot training data
    plot_data(X, y)
    plt.show()

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
#
def part7_training_SVM_with_RBF_Kernel_dataset3():
    # Load from ex6data3:
    # You will have X, y in your environment
    data = scipy.io.loadmat('ex6data3.mat')   
    X = data['X']
    y = data['y'].flatten()
    Xval = data['Xval']
    yval = data['yval'].flatten()
    # Try different SVM Parameters here
    C, sigma = dataset3_params(X, y, Xval, yval)
    gamma = 1.0 / (2.0*sigma**2)
    # Train the SVM

    clf = svm.SVC(C=C, kernel='rbf', tol=1e-3, gamma=gamma)
    model = clf.fit(X, y)
    visualize_boundary(X, y, model)
    plt.show()

def dataset3_params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3
    C_list =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    error_least = np.size(y, 0)
    for C_i in C_list:
        for sigma_i in sigma_list:
            clf = svm.SVC(C=C_i, kernel='rbf', tol=1e-3, gamma=1/(2*sigma_i**2))
            model = clf.fit(X, y)
            prediction = model.predict(Xval)
            error = np.mean(np.double(prediction != yval))
            if error_least > error:
                error_least = error
                C = C_i
                sigma = sigma_i  
    return C, sigma

def main():
    #part1_loading_and_visualizing_data()
    part2_training_linear_SVM()
    part3_implementing_Gaussian_Kernel()
    part4_visualizing_dataset2()
    part5_training_SVM_with_RBF_Kernel_dataset2()
    part6_visualizing_dataset3()
    part7_training_SVM_with_RBF_Kernel_dataset3()

if __name__ == "__main__":
    main()