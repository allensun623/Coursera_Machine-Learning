# Logistic Regression
from matplotlib import use

use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     gradientFunction.py
#     predict.py
#     costFunctionReg.py
#     gradientFunctionReg.py
#     n.b. This files differ in number from the Octave version of ex2.
#          This is due to the scipy optimization taking only scalar
#          functions where fmiunc in Octave takes functions returning
#          multiple values.
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.



    # ==================== Part 1: Plotting ====================
def part1_plotting(X, y ):
    print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")
    plotData(X, y)

def plotData(X, y):
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = y
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x1[pos], x2[pos], marker='P', c='b')
    p2 = plt.scatter(x1[neg], x2[neg], marker='o', c='y')
    plt.legend((p1, p2), 
                ['Admitted', 'Not admitted'], 
                loc='upper right', 
                fontsize='x-large', 
                shadow=True,
                numpoints=1,
                framealpha=0.8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

    # # ============ Part 2: Compute Cost and Gradient ============
    # #  Setup the data matrix appropriately, and add ones for the intercept term
def part2_ComputeCostAndGradient(X, y):    
    m, n = X.shape

    # Add intercept term to x and X_test
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    # Compute and display initial cost and gradient
    cost = costFunction(initial_theta, X, y)
    print("Cost at initial theta (zeros): %f" % cost)

    grad = gradientFunction(initial_theta, X, y)
    print("Gradient at initial theta (zeros): " + str(grad))
    
    return m, n, X, initial_theta

def sigmoid(initial_theta, X):
    z = -X.dot(initial_theta)
    g = 1 / (1+np.exp(z))
    return g


def costFunction(initial_theta, X, y):
    #initial cost
    m = np.size(X, axis=0)
    h = sigmoid(initial_theta, X)    
    J = -1 / m * (y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h)))
    return J

def gradientFunction(initial_theta, X, y):
    #initial gradient
    m = np.size(X, axis=0)
    h = sigmoid(initial_theta, X)
    grad = 1 / m * (h-y).T.dot(X)
    return grad

    # ============= Part 3: Optimizing using scipy  =============
def part3_OptimizingUsingScipy(X, y):
    m, n, X, initial_theta = part2_ComputeCostAndGradient(X, y)
    res = minimize(costFunction, initial_theta, method='TNC',
                jac=gradientFunction, args=(X, y), options={'gtol': 1e-3, 'disp': True, 'maxiter': 1000})

    theta = res.x
    cost = res.fun
    #result = op.minimize(costFunction, x0=init_theta, method='BFGS', jac=gradFunction, args=(X, Y))
    #theta = result.x
    # Print theta to screen
    print("Cost at theta found by scipy: %f" % cost)
    print("theta: ", theta)
    # Plot Boundary
    plotDecisionBoundary(m, theta, X, y)
    return X, theta

def plotDecisionBoundary(m, theta, X, y):
    x1 = X[:, 1]
    x2 = X[:, 2]
    y = y
    pos = np.where(y==1)
    neg = np.where(y==0)
    p1 = plt.scatter(x1[pos], x2[pos], marker='P', c='b')
    p2 = plt.scatter(x1[neg], x2[neg], marker='o', c='y')
    plot_x = np.array([np.min(x1)-2, np.max(x1)+2])
    print("plot_x: ", plot_x)
    plot_y = -1/theta[2]*(theta[1]*plot_x+theta[0])
    print("plot_y: ", plot_y)
    plt.plot(plot_x, plot_y)
    # Labels and Legend
    plt.legend((p1, p2), 
                ['Admitted', 'Not admitted'], 
                loc='upper right', 
                fontsize='x-large', 
                shadow=True,
                numpoints=1,
                framealpha=0.8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

    #  ============== Part 4: Predict and Accuracies ==============
def part4_PredictAndAccuracies(X,y):
    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2
    X, theta = part3_OptimizingUsingScipy(X, y)
    prob = sigmoid(np.array([1, 45, 85]), theta)
    print("For a student with scores 45 and 85, we predict an admission probability of %f" % prob)
    # Compute accuracy on our training set
    p = predict(theta, X)
    acc = 1.0*np.where(p == y)[0].size/len(p) * 100
    print("Train Accuracy: %f "% acc)

def predict(theta, X):
    m = np.size(X, axis=0)
    p = np.zeros((m,))
    pos = np.where(X.dot(theta) >= 0)
    neg = np.where(X.dot(theta) < 0)
    p[pos] = 1
    p[neg] = 0
    return p



def main():
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    #part1_plotting(X, y)
    #part2_ComputeCostAndGradient(X, y)    
    #part3_OptimizingUsingScipy(X, y)
    part4_PredictAndAccuracies(X, y)

if __name__ == "__main__":
    main()
