## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
import scipy.io
import numpy as np
from matplotlib import use
use('TkAgg')
import display_data as display
from scipy.optimize import minimize


## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#
def part1_loading_and_visualizing_data(X, y):
    # Load Training Data
    print("Loading and Visualizing Data ...")

    m, _ = X.shape

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(range(m))
    sel = X[rand_indices[0:100], :]

    display.display_data(sel, X)



## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#
def part2_vectorize_logistic_regression(X, y):
    print("Training One-vs-All Logistic Regression...")
    num_labels = 10          # 10 labels, from 1 to 10
                            # (note that we have mapped "0" to label 10)
    Lambda = 0.1
    all_theta = one_vs_all(X, y, num_labels, Lambda)
    return all_theta

def one_vs_all(X, y, num_labels, Lambda):
    # Some useful variables
    m, n = X.shape
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))
    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))
    # Set Initial theta
    initial_theta = np.zeros((n + 1, 1))

    # This function will return theta
    for i in range(num_labels):
        num = 10 if i == 0 else i
        result = minimize(cost_function_reg, initial_theta, method='BFGS'\
                 ,jac=gradient_function_reg, args=(X, 1*(y == num), Lambda), options={'maxiter': 50})
        all_theta[i, :] = result.x

    return X, all_theta

def sigmoid(z):
    g = 1/ (1 + np.exp(-z))
    return g

def gradient_function_reg(theta, x, y, Lambda):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    grad = np.zeros(np.size(theta))
    grad[0] = 1 / m * (x[:, 0].dot(h-y))
    grad[1:] = 1/m*(x[:, 1:].T.dot(h-y))+Lambda/m*theta[1:]
    return grad

def cost_function_reg(theta, X, y, Lambda):
    m, _ = X.shape
    h = sigmoid(X.dot(theta))
    lrJ = -1/m*(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h))) + Lambda/(2*m)*theta[1:].T.dot(theta[1:]) 
    return lrJ

## ================ Part 3: Predict for One-Vs-All ================
def part3_predict_one_vs_all(X, y):    
    #  After ...
    X, all_theta = part2_vectorize_logistic_regression(X, y)
    pred = predict_one_vs_all(all_theta, X)
    print("Training Set Accuracy: %f%%" % (np.mean(pred==(y%10))*100))

def predict_one_vs_all(all_theta, X):
    pred = np.argmax((X.dot(all_theta.T)), axis=1)
    return pred

def main():

    data = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
    X = data['X']
    y = data['y'][:, 0]
    #part1_loading_and_visualizing_data(X, y)
    #part2_vectorize_logistic_regression(X, y)
    part3_predict_one_vs_all(X, y)

if __name__ == "__main__":
    main()