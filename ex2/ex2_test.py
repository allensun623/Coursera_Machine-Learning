# -*- coding: utf-8 -*-

from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import plot_decision_boundary as plt_booundry

    # ==================== Part 1: Plotting ====================
def part1_plotting(X, y):
    print("Plot data...")
    plotData(X, y)

def plotData(X, y):
    m, n = X.shape
    x1 = X[:, n-2]
    x2 = X[:, n-1]
    pos = np.where(y==1)
    neg = np.where(y==0)
    p1 = plt.scatter(x1[pos], x2[pos], c='black', marker='+')
    p2 = plt.scatter(x1[neg], x2[neg], c='y', marker='o')
    plt.legend((p1, p2), 
                ['Admitted', 'Not admitted'], 
                loc='upper right', 
                fontsize='x-large', 
                shadow=True,
                numpoints=1,
                framealpha=0.8)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.show()
    
    # # ============ Part 2: Compute Cost and Gradient ============
    # #  Setup the data matrix appropriately, and add ones for the intercept term
def part2_ComputeCostAndGradient(X, y):    
    m, n = X.shape
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    init_theta = np.zeros(np.size(X, axis=1))
    cost = compute_cost(init_theta, X, y)
    grad = gradient(init_theta, X, y)
    print("Cost at initial theta (zeros): ", cost)
    print("Gradient at initial theta (zeros): " + str(grad))
    return m, n, X, init_theta

def sigmoid(theta, X):
    z = X.dot(theta)
    g = 1 / (1+np.exp(-z))
    return g 

def compute_cost(init_theta, X, y):
    m, n = X.shape 
    h = sigmoid(init_theta, X)
    J = -1 / m * (y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h)))
    return J

def gradient(init_theta, X, y):
    m, n = X.shape 
    h = sigmoid(init_theta, X)
    g = 1/m*(h-y).T.dot(X)
    return g

    # ============= Part 3: Optimizing using scipy  =============
def part3_optimizing_using_scipy(X, y):
    m, n, X, initial_theta = part2_ComputeCostAndGradient(X, y)
    res = minimize(compute_cost, initial_theta, method='TNC',
                jac=gradient, args=(X, y), options={'gtol': 1e-3, 'disp': True, 'maxiter': 1000})

    theta = res.x
    cost = res.fun
    # Print theta to screen
    print("Cost at theta found by scipy: %f" % cost)
    print("theta: ", theta)
    plt_booundry.plot_decision_boundary(theta, X, y)
    plotData(X, y)
    return m, X, theta
  
    #  ============== Part 4: Predict and Accuracies ==============
def part4_predict_and_accuracies(X,y):
    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2
    m, X, theta = part3_optimizing_using_scipy(X, y)
    prob = sigmoid(theta, np.asarray([1, 45, 85]))
    print("For a student with scores 45 and 85, we predict an admission probability of %f" % prob)
    accuracy = predict(theta, X, y, m)
    print("Train Accuracy: %f "% accuracy)
    
def predict(theta, X, y, m):
    training_y = X.dot(theta)
    pos = np.where(training_y>=0.5)
    result_traning_y = np.zeros((m,))  
    result_traning_y[pos] = 1
    accuracy = np.size(np.where(result_traning_y==y)) / m *100
    return accuracy

def main():
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = data[:, 0: 2]
    y = data[:, 2]
    #part1_plotting(X, y)
    #part2_ComputeCostAndGradient(X, y)
    #part3_optimizing_using_scipy(X, y)
    part4_predict_and_accuracies(X,y)

if __name__ == "__main__":
    main()