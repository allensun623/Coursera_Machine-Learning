## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import scipy.io
from scipy.optimize import minimize
import display_data as display



## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#
def part1_loading_and_visualizing_data(X, y):

    m, _ = X.shape

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(range(m))
    sel = X[rand_indices[0:100], :]

    display.display_data(sel, X)



## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

def part2_loading_parameters(X, y):
    # Load the weights into variables Theta1 and Theta2
    thetainfo = scipy.io.loadmat('ex4weights.mat')
    theta1 = thetainfo['Theta1']
    theta2 = thetainfo['Theta2']
    nn_params = np.hstack((theta1.ravel(), theta2.ravel()))
    return nn_params
## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
def part3_compute_cost(X, y, input_layer_size, hidden_layer_size, num_labels):
    nn_params = part2_loading_parameters(X, y)
    print("Feedforward Using Neural Network ...")

    # Weight regularization parameter (we set this to 0 here).
    Lambda = 0

    J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
        num_labels, X, y, Lambda)
    print("Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)\n" % J)

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
        num_labels, X, y, Lambda):
    Theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1),], (hidden_layer_size, input_layer_size+1)) # 25 * 401
    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):,], (num_labels, hidden_layer_size+1)) # 10 * 26
    m, _ = np.shape(X)
    a1 = np.column_stack((np.ones((m, 1)), X))
    z2 = a1.dot(Theta1.T)
    a2 = np.column_stack((np.ones((m, 1)), sigmoid(z2)))
    a3 = sigmoid(a2.dot(Theta2.T))
    y_K = np.zeros((m, num_labels))
    y_K[np.arange(m), y-1] = 1
    reg_cost = np.sum(Theta1[:, 1:]*Theta1[:, 1:]) +  np.sum(Theta2[:, 1:]*Theta2[:, 1:])
    J = -1 / m * (np.sum(y_K*np.log(a3) + (1-y_K)*(np.log(1-a3)))) + \
        Lambda / (2*m) * reg_cost
    return J, a3

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

def part4_implement_regularization(X, y, input_layer_size, hidden_layer_size, num_labels):
    print("Checking Cost Function (w/ Regularization) ...")
    nn_params = part2_loading_parameters(X, y)
    # Weight regularization parameter (we set this to 1 here).
    Lambda = 1

    J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

    print("Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)" % J)



## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

def part5_sigmoid_gradient(X, y):
    print("Evaluating sigmoid gradient...")

    g = sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]))
    print("Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]: ")
    print(g)

def sigmoid_gradient(z):
    g_prime = sigmoid(z)*(1-sigmoid(z))
    return g_prime

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

def part6_initializing_pameters(X, y, input_layer_size, hidden_layer_size, num_labels):
    print("Initializing Neural Network Parameters ...")

    initial_Theta1 = rand_initialize_weights(input_layer_size+1, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size+1, num_labels)

    # Unroll parameters
    initial_nn_params = np.hstack((initial_Theta1.ravel(), initial_Theta2.ravel()))
    return initial_nn_params

def rand_initialize_weights(input_layer, output_layer):
    epsilon = 0.12
    theta = np.random.rand(input_layer, output_layer) * 2*epsilon - epsilon 
    return theta


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#

def part7_implement_backpropagation(X, y, input_layer_size, hidden_layer_size, num_labels):
    print("Checking Backpropagation... ")
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Lambda = 1
    nn_params = part2_loading_parameters(X, y)
    Delta = backpropagation(nn_params, input_layer_size, hidden_layer_size, 
                            num_labels, X, y, Lambda)
    numDelta = computeNumericalGradient(nn_cost_function, nn_params,\
                                       (input_layer_size, hidden_layer_size, num_labels, X, y, Lambda))
    print(Delta, "\n", numDelta)
    #  Check gradients by running ckeck_nn_gradients
    ckeck_nn_gradients()

def backpropagation(nn_params, input_layer_size, hidden_layer_size, 
                    num_labels, X, y, Lambda):
    Theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1),], (hidden_layer_size, input_layer_size+1)) # 25 * 401
    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):,], (num_labels, hidden_layer_size+1)) # 10 * 26
    m, _ = np.shape(X)
    a1 = np.column_stack((np.ones((m, 1)), X))
    z2 = a1.dot(Theta1.T)
    a2 = np.column_stack((np.ones((m, 1)), sigmoid(z2)))
    a3 = sigmoid(a2.dot(Theta2.T))
    y_K = np.zeros((m, num_labels))
    y_K[np.arange(m), y-1] = 1
    #step4
    delta3 = a3 - y_K
    delta2 = delta3.dot(Theta2) * \
            sigmoid_gradient(np.column_stack((np.ones((m,1)), z2)))
    #step5
    Delta2 = delta3.T.dot(a2)
    Delta1 = delta2[:, 1:].T.dot(a1)
    Delta2 = Delta2 / m
    Delta1 = Delta1 / m 
    Delta2[:, 1:] = Delta2[:, 1:] + Lambda*Theta2[:, 1:]/m
    Delta1[:, 1:] = Delta1[:, 1:] + Lambda*Theta1[:, 1:]/m
    Delta = np.concatenate((Delta1.flatten(), Delta2.flatten()))
    return Delta

def computeNumericalGradient(J, theta, args):
    numgrad = np.zeros(np.size(theta))
    perturb = np.zeros(np.size(theta))
    epsilon = 1e-4
    for i in range(np.size(theta)):
        perturb[i] = epsilon
        loss1, _ = J(theta-perturb, *args)
        loss2, _ = J(theta+perturb, *args)
        numgrad[i] = (loss2-loss1)/(2*epsilon)
        perturb[i] = 0
    return numgrad

def ckeck_nn_gradients():
    epsilon_derivative = 1e-4


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

def part8_implement_regularization(X, y):
    print("Checking Backpropagation (w/ Regularization) ... ")

    #  Check gradients by running checkNNGradients
    Lambda = 3.0
    checkNNGradients(Lambda)

    # Also output the costFunction debugging values
    debug_J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

    print("Cost at (fixed) debugging parameters (w/ lambda = 10): %f (this value should be about 0.576051)\n\n" % debug_J)



## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#

def part8_training_NN(X, y):
    print("Training Neural Network... ")

    #  After you have completed the assignment, change the MaxIter to a larger
    #  value to see how more training helps.
    # options = optimset('MaxIter', 50)

    #  You should also try different values of lambda
    Lambda = 1

    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[0]
    gradFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[1]

    result = minimize(costFunc, initial_nn_params, method='CG', jac=gradFunc, options={'disp': True, 'maxiter': 50.0})
    nn_params = result.x
    cost = result.fun

    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, input_layer_size + 1), order='F').copy()
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                    (num_labels, (hidden_layer_size + 1)), order='F').copy()



## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

def part9_visualize_weight(X, y):

    print("Visualizing Neural Network... ")

    display.display_data(Theta1[:, 1:], X)


## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

def part10_implement_predict(X, y):
    pred = predict(Theta1, Theta2, X)

    accuracy = np.mean(np.double(pred == y)) * 100
    print("Training Set Accuracy: %f\n"% accuracy)


def main():
        # Load Training Data
    print("Loading and Visualizing Data ...")

    data = scipy.io.loadmat('ex4data1.mat')
    X = data['X']
    y = data['y']  # y = [1, 2, ..., 9, 10]
    y = np.squeeze(y)
    ## Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10   
                            # (note that we have mapped "0" to label 10)

    #part1_loading_and_visualizing_data(X, y)
    #part2_loading_parameters(X, y)
    #part3_compute_cost(X, y, input_layer_size, hidden_layer_size, num_labels)
    #part4_implement_regularization(X, y, input_layer_size, hidden_layer_size, num_labels)
    #part5_sigmoid_gradient(X, y)
    #part6_initializing_pameters(X, y, input_layer_size, hidden_layer_size, num_labels)
    part7_implement_backpropagation(X, y, input_layer_size, hidden_layer_size, num_labels)

if __name__ == "__main__":
    main()