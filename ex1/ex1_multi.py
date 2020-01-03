from matplotlib import use
use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# ================ Part 1: Feature Normalization ================

def part1_FeatureNormalization():
    print("Loading data ...")

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = y.T.size


    # Print out some data points
    print("First 10 examples from the dataset:")
    print(np.column_stack((X[:10], y[:10])))

    # Scale features and set them to zero mean
    print("Normalizing Features ...")

    X, mu, sigma = featureNormalize(X)
    print("[mu] [sigma]")
    print(mu, sigma)

    # Add intercept term to X
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    return X, y

def featureNormalize(x):
    mu = np.average(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = np.divide(x-mu, sigma)
    return x_norm, mu, sigma
# ================ Part 2: Gradient Descent ================
#
# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

def part2_GradientDescent():
    print("Running gradient descent ...")
    X, y = part1_FeatureNormalization()
    # Choose some alpha value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent 
    theta = np.zeros(3)
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.plot(J_history, '-b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # Display gradient descent's result
    print("Theta computed from gradient descent: ")
    print(theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.array([1, 1650, 3]).dot(theta)

    print("Predicted price of a 1650 sq-ft, 3 br house")
    print("(using gradient descent):" )
    print(price)

def computeCost(X, y, theta):
    #J = 1 / 2m * sum((h-y)^2)
    #h = sum(theta_i*x_i)
    m = np.size(y, 0)  # number of training examples
    h = X.dot(theta) # vector of hyposis
    delta_sqr = np.sum(np.square(h - y ))
    J = 1 / (2*m) * delta_sqr
    return J

def gradientDescentMulti(X, y, theta, alpha, iterations):
    #theta_j: = theta_j - alpha * deltaJ
    #deltaJ = (1/m)*sum((h-y)*xi_j)
    #h = sum(theta_i*x_i)
    J_history = []
    m = y.size  # number of training examples
    
    for i in range(iterations):
        deltaJ = ((X.dot(theta) - y)).T.dot(X) / m        
        theta = theta - alpha * deltaJ
        J_history.append(computeCost(X, y, theta))    
    return theta, J_history


# ================ Part 3: Normal Equations ================

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#
def part3_NormalEquations():
    print("Solving with normal equations...")

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = y.T.size

    # Add intercept term to X
    X = np.concatenate((np.ones((m,1)), X), axis=1)

    # Calculate the parameters from the normal equation
    theta = normalEqn(X, y)

    # Display normal equation's result
    print("Theta computed from the normal equations:")
    print("%s \n" % theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.array([1, 1650, 3]).dot(theta)

    # ============================================================

    print("Predicted price of a 1650 sq-ft, 3 br house ")
    print("(using normal equations):\n $%f\n" % price)

def normalEqn(X, y):

    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# ============================================================

def main():
    #part1_FeatureNormalization()
    #part2_GradientDescent()
    part3_NormalEquations()

if __name__ == "__main__":
    main()

