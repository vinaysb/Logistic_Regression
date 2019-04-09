# Logistic Regression

This is a module for logistic regression in python that is modelled based on the Coursera course on machine learning, taught by Andrew Ng.


## Working

It uses BFGS algorithm for minmizing the cost function J (like fminunc in octave).

## How to use

1. Split your testing and training data using test_train_split from sklearn for uniform and random splitting of your data.
2. Pass the Training freatures and labels and the test features to the Logistic_Regression function.
3. Check the accuracy of the prediction using accuracy score from the sklearn module.
