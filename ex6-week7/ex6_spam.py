## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io
from sklearn import svm
from collections import OrderedDict
import string
import re
import pysnooper
import timeit
from nltk.stem import PorterStemmer

## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

def part1_email_preprocessing():
    print("Preprocessing sample email (emailSample1.txt)")

    # Extract Features
    file = open('emailSample1.txt', 'r')
    file_contents = file.readlines()
    word_indices  = process_email(''.join(file_contents))

    # Print Stats
    print("Word Indices: ")
    print(word_indices)

def process_email(file_contents):
    print(file_contents)
    lowercase = file_contents.lower()
    start0 = timeit.timeit()
    words = [filter(item) for item in lowercase.split() if filter(item)]
    #remove_punctuation = words.translate(str.maketrans('', '', string.punctuation))
    #remove_numb = ''.join([i for i in remove_punctuation if not i.isdigit()])
    print(words)
    word_indices = ''
    end0 = timeit.timeit()
    print(end0 - start0)
    return word_indices

def filter(item):
    item = PorterStemmer().stem(item)
    item = ''.join('dollar' if i == '$' else i for i in item)
    #replace numbers with 'numb'
    digit_boolean = any(char.isdigit() for char in item)
    item = ''.join([i for i in item if not i.isdigit()])
    if digit_boolean:
        item = item + 'numb'
    #relace @
    if '@' in item:
        return 'emailaddr'
    #relace URL
    elif 'http://' in item:
        return 'httpaddr'
    #relace $
    elif '$' == item:
        return 'dollar'
    elif item in string.punctuation:
        return False
    else:
        return ''.join([i for i in item.translate(str.maketrans('', '', string.punctuation))])


## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.
def part2_feature_extraction():
    print("Extracting features from sample email (emailSample1.txt)")

    # Extract Features
    file = open('emailSample1.txt')
    file_contents = file.readlines()
    word_indices = processEmail(''.join(file_contents))
    features = emailFeatures(word_indices)

    # Print Stats
    print("Length of feature vector: %d"% features.size)
    print("Number of non-zero entries: %d"% sum(features > 0))

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.
def part3_train_linear_SVM_for_spam_classification():
    # Load the Spam Email dataset
    # You will have X, y in your environment
    data = scipy.io.loadmat('spamTrain.mat')
    X = data['X']
    y = data['y'].flatten()

    print("Training Linear SVM (Spam Classification)")
    print("(this may take 1 to 2 minutes) ...")

    C = 0.1
    clf = svm.SVC(C=C, kernel='linear', tol=1e-3, max_iter=200)
    model = clf.fit(X, y)

    p = model.predict(X)

    print("Training Accuracy: %f", np.mean(np.double(p == y)) * 100)

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

def part4_test_spam_classification():
    # Load the test dataset
    # You will have Xtest, ytest in your environment
    data = scipy.io.loadmat('spamTest.mat')
    Xtest = data['Xtest']
    ytest = data['ytest']

    print("Evaluating the trained Linear SVM on a test set ...")

    p = model.predict(Xtest)

    print("Test Accuracy: %f", np. mean(np.double(p == ytest)) * 100)


## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#
def part5_top_predictors_of_spam():
    # Sort the weights and obtain the vocabulary list

    t = sorted(list(enumerate(model.coef_[0])),key=lambda e: e[1], reverse=True)
    d = OrderedDict(t)
    idx = d.keys()
    weight = d.values()
    vocabList = getVocabList()

    print("Top predictors of spam: ")
    for i in range(15):
        print(" %-15s (%f)" %(vocabList[idx[i]], weight[i]))

    print("Program paused. Press enter to continue.")

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned SVM classifier to determine whether the email is Spam or
#  Not Spam

def part6_try_your_own_emails():
    # Set the file to be read in (change this to spamSample2.txt,
    # emailSample1.txt or emailSample2.txt to see different predictions on
    # different emails types). Try your own emails as well!
    filename = 'spamSample1.txt'

    # Read and predict

    file = open(filename)
    file_contents = file.readlines()
    word_indices = processEmail(''.join(file_contents))
    x = emailFeatures(word_indices)
    p = model.predict(x)

    print("Processed %s\n\nSpam Classification: %d" % (filename, p))
    print("(1 indicates spam, 0 indicates not spam)")


def main():
    part1_email_preprocessing()

if __name__ == "__main__":
    main()