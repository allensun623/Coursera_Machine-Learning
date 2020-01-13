## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
import scipy.io
import numpy as np
from matplotlib import use
use('TkAgg')
import display_data as display
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import display_data as display
import math


def part1_loading_and_visualizing_data(X, y):
    # Load Training Data
    print("Loading and Visualizing Data ...")

    m, _ = X.shape

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(range(m))
    sel = X[rand_indices[0:100], :]

    display.display_data(sel, X)

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

def part2_loading_pameters(X, y):
    print("Loading Saved Neural Network Parameters ...")

    # Load the weights into variables Theta1 and Theta2
    data = scipy.io.loadmat('ex3weights.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']
    return Theta1, Theta2

## ============ Part 3: Implement Predict with Neural Network =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
def part3_implement(X, y):
    Theta1, Theta2 =  part2_loading_pameters(X, y)
    m, _ = X.shape
    pred = predict(Theta1, Theta2, X)

    print("Training Set Accuracy: %f\n", (np.mean(np.double(pred == np.squeeze(y))) * 100))

    #  To give you an idea of the network's output, you can also run
    #  through the examples one at the a time to see what it is predicting.
    #  Randomly permute examples
    num = 10
    rindex = np.random.permutation(m)
    for i in range(num):
        print('Displaying Example Image')
        displayData(X[rindex[i]:rindex[i]+1, :], X)
        pred = predict(Theta1, Theta2, X[rindex[i]:rindex[i]+1, :])
        print('Neural Network Prediction: %d (digit %d)' % (pred, pred % 10))
        _ = input('Press [Enter] to continue.')

def predict(Theta1, Theta2, X):
    m, _ = np.shape(X)
    a1 = np.column_stack((np.ones((m, 1)), X))
    a2_0 = sigmoid(a1.dot(Theta1.T))
    a2 = np.column_stack((np.ones((m, 1)), a2_0))
    a3 = sigmoid(a2.dot(Theta2.T))
    pred = np.argmax(a3, axis=1) + 1
    return pred

def sigmoid(z):
    g = 1 / (1+np.exp(-z))
    return g

def displayData(x, X):
    width = round(math.sqrt(np.size(x, 1)))
    m, n = np.shape(x)
    height = int(n/width)
    # show image numbers
    drows = math.floor(math.sqrt(m))
    dcols = math.ceil(m/drows)

    pad = 1
    # blank canvas
    darray = -1*np.ones((pad+drows*(height+pad), pad+dcols*(width+pad)))

    curr_ex = 0
    for j in range(drows):
        for i in range(dcols):
            if curr_ex >= m:
                break
            max_val = np.max(np.abs(X[curr_ex, :]))
            darray[pad+j*(height+pad):pad+j*(height+pad)+height, pad+i*(width+pad):pad+i*(width+pad)+width]\
                = x[curr_ex, :].reshape((height, width))/max_val
            curr_ex += 1
        if curr_ex >= m:
            break

    plt.imshow(darray.T, cmap='gray')
    plt.show()

def main():

    ## Setup the parameters you will use for this part of the exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25
    num_labels = 10
    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset. 
    #  You will be working with a dataset that contains handwritten digits.
    #
    data = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
    X = data['X']
    y = data['y'][:, 0]
    part3_implement(X, y)


if __name__ == "__main__":
    main()
