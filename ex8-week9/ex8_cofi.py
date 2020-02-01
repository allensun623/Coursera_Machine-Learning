## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#

from matplotlib import use, cm
use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import minimize

## =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.

def part1_loading_movie_ratings_dataset():
    print("Loading movie ratings dataset.")

    #  Load data
    data = scipy.io.loadmat('ex8_movies.mat')
    Y = data['Y']
    R = data['R'].astype(bool)
    #  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
    #  943 users
    #
    #  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    #  rating to movie i

    #  From the matrix, we can compute statistics like average rating.
    print("Average rating for movie 1 (Toy Story): %f / 5" % np.mean(Y[0, R[0, :]]))

    #  We can "visualize" the ratings matrix by plotting it with imagesc

    plt.figure()
    plt.imshow(Y, aspect='equal', origin='upper', extent=(0, Y.shape[1], 0, Y.shape[0]/2.0))
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.show()


## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.

def part2_collaborative_filtering_cost_function():
    #  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    data = scipy.io.loadmat('ex8_movies.mat')
    Y = data['Y']
    R = data['R'].astype(bool)

    data = scipy.io.loadmat('ex8_movieParams.mat')
    X = data['X']
    Theta = data['Theta']
    num_users = data['num_users']
    num_movies = data['num_movies']
    num_features = data['num_features']

    #  Reduce the data set size so that this runs faster
    num_users = 4
    num_movies = 5
    num_features = 3
    X = X[:num_movies, :num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies, :num_users]
    R = R[:num_movies, :num_users]
    params = np.hstack((X.flatten(), Theta.flatten()))
    #  Evaluate cost function
    Lambda = 0.0
    Alpha = 1.0 #learning rate
    J, grad = cofi_cost_func(params, Y, R, num_users, num_movies,
                num_features, Lambda, Alpha)
            
    print("Cost at loaded parameters: %f \n(this value should be about 22.22)" % J)
    return params, Y, R, num_users, num_movies, num_features

def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, Lambda, Alpha):
    X = np.array(params[:num_movies*num_features].reshape(num_movies, num_features)).copy()
    Theta = np.array(params[num_movies*num_features:].reshape(num_users, num_features)).copy()
    # You need to return the following values correctly
    J = 0.0
    h = X.dot(Theta.T)
    J = 1/2*np.sum(((h-Y)*R)**2) + \
        Lambda/2*(np.sum(Theta**2)+np.sum(X**2))
    X_grad = Alpha * (((h-Y)*R).dot(Theta)+Lambda*X)
    Theta_grad = Alpha * (((h-Y)*R).T.dot(X)+Lambda*Theta)
    grad = np.hstack((X_grad.flatten(), Theta_grad.flatten()))
    return J, grad

## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement 
#  the collaborative filtering gradient function. Specifically, you should 
#  complete the code in cofiCostFunc.m to return the grad argument.
#  
def part3_collaborative_filtering_gradient():
    print("Checking Gradients (without regularization) ...")
    Lambda = 0.0
    Alpha = 1.0
    #  Check gradients by running check_nn_gradients
    check_cost_function(Lambda, Alpha)


def check_cost_function(Lambda=0.0, Alpha=1.0):
## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = X_t.dot(Theta_t.T)
    Y[np.where(np.random.random_sample(Y.shape) > 0.5, True, False)] = 0
    R = np.zeros(Y.shape)
    R[np.where(Y != 0, True, False)] = 1

    ## Run Gradient Checking
    X = np.random.random_sample(X_t.shape)
    Theta = np.random.random_sample(Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

   # Unroll parameters
    params = np.hstack((X.flatten(), Theta.flatten()))
    numgrad = ckeck_nn_gradients(cofi_cost_func, params, \
                        (Y, R, num_users, num_movies, num_features, Lambda, Alpha))
    _, grad = cofi_cost_func(params, Y, R, num_users, num_movies, num_features, Lambda, Alpha)
    print(np.column_stack((numgrad, grad)))

    print("The above two columns you get should be very similar.\n", \
            "(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n")

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print("If your backpropagation implementation is correct, then\n", \
          "the relative difference will be small (less than 1e-9). \n", \
          "\nRelative Difference: %g\n" % diff)

def ckeck_nn_gradients(J, theta, args):
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

## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for 
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#  
def part4_collaborative_filtering_cost_regularization():
#  Evaluate cost function

    Lambda = 1.5
    Alpha = 0.0
    params, Y, R, num_users, num_movies, num_features = part2_collaborative_filtering_cost_function()
    J, _ = cofi_cost_func(params, Y, R, num_users, num_movies, num_features, Lambda, Alpha)
            
    print("Cost at loaded parameters (lambda = 1.5): %f \n(this value should be about 31.34)\n" % J)


## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement 
#  regularization for the gradient. 
#
def part5_collaborative_filtering_gradient_regularization():
#  

    print("Checking Gradients (with regularization) ...")
    Lambda = 1.5
    Alpha = 1.0
    #  Check gradients by running check_nn_gradients
    check_cost_function(Lambda, Alpha)

## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#
def part6_entering_ratings_for_a_new_user():
    movie_list = load_movie_list()

    #  Initialize my ratings
    my_ratings = np.zeros(1682)

    # Check the file movie_idx.txt for id of each movie in our dataset
    # For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
    my_ratings[0] = 4

    # Or suppose did not enjoy Silence of the Lambs (1991), you can set
    my_ratings[97] = 2

    # We have selected a few movies we liked / did not like and the ratings we
    # gave are as follows:
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5

    print("New user ratings:")
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print("Rated %d for %s\n" % (my_ratings[i], movie_list[i]))
    return movie_list, my_ratings

def load_movie_list():
    with open('movie_ids.txt', 'r+', encoding = "ISO-8859-1") as file:
        data = file.readlines()
    #split(' ', 1)[1] : 
    # splite once, and get the string after the first space
    movies = [item.rstrip('\n').split(' ', 1)[1] for item in data]
    return movies

## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating 
#  dataset of 1682 movies and 943 users
#
def part7_learning_movie_ratings():
    print("\nTraining collaborative filtering...")
    movie_list, my_ratings = part6_entering_ratings_for_a_new_user()
    #  Load data
    data = scipy.io.loadmat('ex8_movies.mat')
    Y = data['Y']
    R = data['R'].astype(bool)

    #  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
    #  943 users
    #
    #  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    #  rating to movie i

    #  Add our own ratings to the data matrix
    Y = np.column_stack((my_ratings, Y))
    R = np.column_stack((my_ratings, R)).astype(bool)

    #  Normalize Ratings
    Ynorm, Ymean = normalize_ratings(Y, R)

    #  Useful Values
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10

    # Set Initial Parameters (Theta, X)
    X = np.random.rand(num_movies, num_features)
    Theta = np.random.rand(num_users, num_features)

    initial_parameters = np.hstack((X.flatten(), Theta.flatten()))
    # Set Regularization
    Lambda = 10
    Alpha = 1.0

    cost_func = lambda p: cofi_cost_func(p, Ynorm, R, num_users, num_movies, num_features, Lambda, Alpha)[0]
    grad_func = lambda p: cofi_cost_func(p, Ynorm, R, num_users, num_movies, num_features, Lambda, Alpha)[1]

    result = minimize(cost_func, initial_parameters, method='CG', jac=grad_func, 
                        options={'disp': True, 'maxiter': 1000.0})
    theta = result.x
    cost = result.fun


    # Unfold the returned theta back into U and W
    X = theta[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = theta[num_movies*num_features:].reshape(num_users, num_features)

    print("Recommender system learning completed.")
    return X, Theta, Ymean, movie_list, my_ratings

def normalize_ratings(Y, R):
    Ymean = np.mean(Y*R, axis=1)
    print(Ymean)
    Ynorm = (Y.T-Ymean.T).T
    return Ynorm, Ymean

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#
def part8_recommendation_for_you():
    X, Theta, Ymean, movie_list, my_ratings = part7_learning_movie_ratings()
    p = X.dot(Theta.T)
    my_predictions = p[:, 0] + Ymean

    # sort predictions descending
    pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])
    post = pre[pre[:,1].argsort()[::-1]]
    r = post[:,1]
    ix = post[:,0]

    print("\nTop recommendations for you:")
    for i in range(10):
        j = int(ix[i])
        print("Predicting rating %.1f for movie %s\n" % (my_predictions[j], movie_list[j]))

    print("\nOriginal ratings provided:")
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print("Rated %d for %s\n" % (my_ratings[i], movie_list[i]))


def main():
    #part1_loading_movie_ratings_dataset()
    #part2_collaborative_filtering_cost_function()
    #part3_collaborative_filtering_gradient()
    #part4_collaborative_filtering_cost_regularization()
    #part5_collaborative_filtering_gradient_regularization()
    #part6_entering_ratings_for_a_new_user()
    #part7_learning_movie_ratings()
    part8_recommendation_for_you()

if __name__ == '__main__':
    main()