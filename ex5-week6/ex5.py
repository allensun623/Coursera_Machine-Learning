
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pysnooper

## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linear_reg_cost_func.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#
def part1_visualizing_data(X, y, Xval, yval, Xtest):


    # Plot training data
    plt.scatter(X, y, marker='x', s=60, color='r', lw=1.5)
    plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
    plt.xlabel('Change in water level (x)')     # Set the x-axis label
    plt.show()

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

def part2_regularized_linear_regression_cost(X, y):
    m = X.size
    theta = np.array([1, 1])
    J = linear_reg_cost_func(np.column_stack((np.ones(m), X)), y, theta, 1)[0]

    print("Cost at theta = [1  1]: %f \n(this value should be about 303.993192)\n" % J)

def linear_reg_cost_func(X, y, theta, Lambda):
    m, _ = X.shape
    h = X.dot(theta)
    J = (h-y).T.dot(h-y)/(2*m) + Lambda/(2*m)*theta[1:].T.dot(theta[1:])
    grad = 1 / m * (h-y).T.dot(X)
    return J, grad


## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#
def part3_regularized_linear_regression_gradient(X, y):
    theta = np.array([1, 1])
    m = X.size
    J, grad = linear_reg_cost_func(np.column_stack((np.ones(m), X)), y, theta, 1)

    print("Gradient at theta = [1  1]:  [%f %f] \n(this value should be about [-15.303016 598.250744])\n" %(grad[0], grad[1]))



## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#
def part4_train_linear_regression(X, y):
    #  Train linear regression with Lambda = 0
    Lambda = 0
    m = X.size
    theta = train_linear_reg(np.column_stack((np.ones(m), X)), y, 1, 'CG', 200)[0]

    #  Plot fit over the data
    plt.scatter(X, y, marker='x', s=20, color='r', lw=1.5)
    plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
    plt.xlabel('Change in water level (x)')     # Set the x-axis label
    plt.plot(X, np.column_stack((np.ones(m), X)).dot(theta), '--', lw=2.0)
    plt.show()

def train_linear_reg(X, y, Lambda, method, maxiter):
    m, n = X.shape
    init_theta = np.zeros((n, 1))
    cost_func = lambda theta: linear_reg_cost_func(X, y, theta, Lambda)[0]
    grad_func = lambda theta: linear_reg_cost_func(X, y, theta, Lambda)[1]
    result = minimize(cost_func, init_theta, method=method, jac=grad_func, options={'disp': True, 'maxiter': maxiter})
    J = result.fun
    theta = result.x
    return J, theta 


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
#
def part5_learning_curve_for_linear_regression(X, y, Xval, yval):
    Lambda = 0
    m = X.size
    error_train, error_val = learning_curve(np.column_stack((np.ones(m), X)), y,
                                        np.column_stack((np.ones(Xval.shape[0]), Xval)), yval, Lambda)
    plt.figure()
    plt.plot(range(m), error_train, color='b', lw=0.5, label='Train')
    plt.plot(range(m), error_val, color='g', lw=0.5, label='Cross Validation')
    plt.title('Learning curve for linear regression')
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')

    plt.xlim(0, 13)
    plt.ylim(0, 150)
    plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
    plt.show()
    print("Training Examples\tTrain Error\tCross Validation Error")
    for i in range(m):
        print("  \t%d\t\t%f\t%f" % (i, error_train[i], error_val[i]))

def learning_curve(X, y, Xval, yval, Lambda):
    m, n = np.shape(X)
    theta = [train_linear_reg(X[: i+1,], y[: i+1,], Lambda, 'CG', 200)[1] for i in range(m)]
    error_train = [linear_reg_cost_func(X[: i+1,], y[: i+1,], theta[i], Lambda)[0] for i in range(m)]
    error_val = [linear_reg_cost_func(Xval, yval, theta[i], Lambda)[0] for i in range(m)]
    return error_train, error_val

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete poly_features to map each example into its powers
#

def part6_feature_mapping_for_polynomial_regression(X, y, Xval, yval, Xtest):
    p = 8
    m = X.size
    # Map X onto Polynomial Features and Normalize
    X_poly = poly_features(X, p)
    X_poly, mu, sigma = feature_normalize(X_poly)  # Normalize
    X_poly = np.column_stack((np.ones(m), X_poly))                   # Add Ones

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = poly_features(Xtest, p)
    X_poly_test = X_poly_test - mu
    X_poly_test = X_poly_test / sigma
    X_poly_test = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_test))        # Add Ones

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = poly_features(Xval, p)
    X_poly_val = X_poly_val - mu
    X_poly_val = X_poly_val / sigma
    X_poly_val = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_val))           # Add Ones

    print("Normalized Training Example 1:")
    print(X_poly[0, :])
    return X_poly, X_poly_val, mu, sigma, p

def poly_features(X, p):
    X_poly = np.asarray([np.power(X, i+1) for i in range(p)]).T
    return X_poly

def feature_normalize(X_poly):
    mu = np.mean(X_poly, 0)
    sigma = np.std(X_poly, 0, ddof=1)
    X_poly = (X_poly-mu) / sigma
    return X_poly, mu, sigma

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of Lambda. The code below runs polynomial regression with 
#  Lambda = 0. You should try running the code with different values of
#  Lambda to see how the fit and learning curve change.
#
def part7_learning_curve_for_polynomial_regression(X, y, Xval, yval, Xtest):
    X_poly, X_poly_val, mu, sigma, p = part6_feature_mapping_for_polynomial_regression(X, y, Xval, yval, Xtest)
    Lambda = 10
    m = X.size
    theta = train_linear_reg(X_poly, y, Lambda, method='BFGS', maxiter=10)[1]

    # Plot training data and fit
    plt.figure()
    plt.scatter(X, y, marker='x', s=10, edgecolor='r', lw=1.5)

    x_simu, y_simu = plot_fit(min(X), max(X), mu, sigma, theta, p)

    plt.scatter(X, y, marker='x', s=20, color='r', lw=1.5)
    plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
    plt.xlabel('Change in water level (x)')     # Set the x-axis label
    plt.plot(x_simu, y_simu, '--', lw=2.0)
    plt.title('Polynomial Regression Fit (Lambda = %f)' % Lambda)
    plt.show()

    error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, Lambda)
    plt.plot(np.arange(m)+1, error_train, label='Train')
    plt.plot(np.arange(m)+1, error_val, label='Cross Validation')
    plt.title('Polynomial Regression Learning Curve (Lambda = %f)' % Lambda)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.xlim(0, 13)
    plt.ylim(0, 150)
    plt.legend()
    plt.show()
    print("Polynomial Regression (Lambda = %f)\n\n" % Lambda)
    print("# Training Examples\tTrain Error\tCross Validation Error")
    for i in range(m):
        print("  \t%d\t\t%f\t%f" % (i, error_train[i], error_val[i]))

def plot_fit(X_min, X_max, mu, sigma, theta, p):
    #  Plot fit over the data
    X = np.arange(X_min-15, X_max+25, 0.05)
    m = np.size(X, 0)
    X_poly = poly_features(X, p)
    X_poly = (X_poly-mu) / sigma
    y = np.column_stack((np.ones((m, 1)), X_poly)).dot(theta)
    return X, y

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  Lambda on a validation set. You will then use this to select the
#  "best" Lambda value.
#
def part8_validation_for_selecting_Lambda(X, y, Xval, yval, Xtest):
    X_poly, X_poly_val, mu, sigma, p = part6_feature_mapping_for_polynomial_regression(X, y, Xval, yval, Xtest)
    
    Lambda_vec, error_train, error_val = validation_curve(X_poly, y, X_poly_val, yval)

    plt.plot(Lambda_vec, error_train, Lambda_vec, error_val)
    plt.legend('Train', 'Cross Validation')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.show()
    print("Lambda\t\tTrain Error\tValidation Error")
    for i in range(np.size(Lambda_vec)):
        print(" %f\t%f\t%f" % (Lambda_vec[i], error_train[i], error_val[i]))


def validation_curve(X_poly, y, X_poly_val, yval):
    m, n = np.shape(X_poly)
    print(X_poly_val.shape)
    Lambda_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    """
    error_train = []
    error_val = []
    for Lambda in Lambda_list:
        theta = train_linear_reg(X_poly, y, Lambda, 'CG', 200)[1]
        print("theta: ")
        print(theta)
        Jtrain = linear_reg_cost_func(X_poly, y, theta, Lambda)[0]
        Jval = linear_reg_cost_func(X_poly_val, yval, theta, Lambda)[0]
        error_train.append(Jtrain)
        error_val.append(Jval)
    """
    theta = [train_linear_reg(X_poly, y, Lambda, 'CG', 200)[1] for Lambda in Lambda_list]
    error_train = [linear_reg_cost_func(X_poly, y, theta[i], Lambda_list[i])[0] for i in range(len(Lambda_list))]
    error_val = [linear_reg_cost_func(X_poly_val, yval, theta[i], Lambda_list[i])[0] for i in range(len(Lambda_list))]
    return np.asarray(Lambda_list), np.asarray(error_train), np.asarray(error_val)

def main():
    # Load Training Data
    print("Loading and Visualizing Data ...")

    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = scipy.io.loadmat('ex5data1.mat')

    # m = Number of examples
    X = data['X'][:, 0]
    y = data['y'][:, 0]
    Xval = data['Xval'][:, 0]
    yval = data['yval'][:, 0]
    Xtest = data['Xtest'][:, 0]
    #part1_visualizing_data(X, y, Xval, yval, Xtest)
    #part2_regularized_linear_regression_cost(X, y)
    #part3_regularized_linear_regression_gradient(X, y)
    #part4_train_linear_regression(X, y)
    #part5_learning_curve_for_linear_regression(X, y, Xval, yval)
    #part6_feature_mapping_for_polynomial_regression(X, y, Xval, yval, Xtest)
    #part7_learning_curve_for_polynomial_regression(X, y, Xval, yval, Xtest)
    part8_validation_for_selecting_Lambda(X, y, Xval, yval, Xtest)

if __name__ == "__main__":
    main()