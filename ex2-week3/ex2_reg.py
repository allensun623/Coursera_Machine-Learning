# Logistic Regression
from matplotlib import use

use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pandas as pd

from ml import mapFeature, mapFeature2, plotData, plotDecisionBoundary

# =========== Part 1: Regularized Logistic Regression ============
def part1_regularized_logistic_regression(X, y):
    #plot 
    plot_data(X, y)

    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = map_feature(X[:, 0], X[:, 1])
    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])
    print(initial_theta)
    # Set regularization parameter lambda to 1
    Lambda = 0.0

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost = cost_function_reg(initial_theta, X, y, Lambda)

    print("Cost at initial theta (zeros): %f" % cost)


def plot_data(X, y):  
    m, n = X.shape
    x_axis = X[:, n-2]
    y_axis = X[:, n-1]
    pos = np.where(y==1)
    neg = np.where(y==0)
    p1 = plt.scatter(x_axis[pos], y_axis[pos], c='black', marker='+')
    p2 = plt.scatter(x_axis[neg], y_axis[neg], c='y', marker='o')
    # Labels and Legend
    plt.legend((p1, p2), 
                ['y=1', 'y=0'],
                loc='upper right', 
                fontsize='x-large', 
                shadow=True,
                numpoints=1,
                framealpha=0.8)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()


def sigmoid(z):
    g = 1 / (1+np.exp(-z))
    return g

def cost_function_reg(theta, X, y, Lambda):
    m, n =  X.shape
    h = sigmoid(X.dot(theta))
    J = -1 / m * (y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h))) + Lambda/2/m*theta.T.dot(theta)
    return J
    
def gradient_function_reg(theta, X, y, Lambda):
    m, n =  X.shape
    h = sigmoid(X.dot(theta))
    g = 1 / m * ((h-y).T.dot(X) + Lambda*theta)
    return g

# ============= Part 2: Regularization and Accuracies =============

def part2_regularization_and_accuracies(X, y):  
    X = map_feature(X[:, 0], X[:, 1])
    initial_theta = np.zeros(X.shape[1])
    # Optimize and plot boundary

    Lambda = 0.0
    result = optimize(X, y, initial_theta, Lambda)
    theta = result.x
    cost = result.fun

    # Print to screen
    print("lambda = " + str(Lambda))
    print("Cost at theta found by scipy: %f" % cost)
    print("theta:", ["%0.4f" % i for i in theta])


    plot_boundary(Lambda, theta, X, y)

    # Compute accuracy on our training set
    p = np.round(sigmoid(X.dot(theta)))
    acc = np.mean(np.where(p == y,1,0)) * 100
    print("Train Accuracy: %f" % acc)
    return result, Lambda, theta, X, y

def map_feature(x1, x2, degree=6):
    X_array = np.array([x1**(i-j) * x2**j for i in range(1,degree+1) for j in range(i+1)])
    X = np.concatenate((np.ones((np.size(x1), 1)), X_array.T), axis=1)
    return X


def optimize(X, y, initial_theta, Lambda):
    result = minimize(cost_function_reg, initial_theta, method='L-BFGS-B',
               jac=gradient_function_reg, args=(X, y, Lambda),
               options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

    return result


# Plot Boundary
def plot_boundary(Lambda, theta, X, y):
    plot_decision_boundary(theta, X, y)
    plt.title(r'$\lambda$ = ' + str(Lambda))

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()

def plot_decision_boundary(theta, x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', s=60, color='r')
    p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', s=60, color='y')
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((np.size(u, 0), np.size(v, 0)))
    for i in range(np.size(u, 0)):
        for j in range(np.size(v, 0)):
            z[i, j] = map_feature(np.array([u[i]]), np.array([v[j]])).dot(theta)
    z = z.T
    [um, vm] = np.meshgrid(u, v)
    plt.contour(um, vm, z, levels=[0], lw=2)
    plt.legend((p1, p2), ('y=1', 'y=0'), loc='upper right', fontsize=8)

# ============= Part 3: Optional Exercises =============
#plot data with various lambda
def part3_optional_excercises(X, y):
    result, Lambda, theta, X, y = part2_regularization_and_accuracies(X, y)
    for Lambda in np.arange(0.0,10,1.0):
        theta = result.x
        print("lambda = " + str(Lambda))
        print("theta: ", ["%0.4f" % i for i in theta])
        plot_boundary(Lambda, theta, X, y)


def main():
    # Initialization

    # Load Data
    #  The first two columns contains the X values and the third column
    #  contains the label (y).

    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = data[:, 0: 2]
    y = data[:, 2]
    #part1_regularized_logistic_regression(X, y)
    part2_regularization_and_accuracies(X, y)
    #part3_optional_excercises(X, y)

if __name__ == "__main__":
    main()