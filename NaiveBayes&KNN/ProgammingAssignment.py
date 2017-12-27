# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:11:57 2017

@author: Yonarp
"""

import numpy as np
import NaiveBayes
import KNearestNeighbours
from sklearn.decomposition import PCA
import PCA as libpca
testErrors = []
trainErrors = []


def main():
    global testErrors
    global trainErrors
    print ("You have 4 choices over four questions, select 1,2,3,4")
    flag = True
    while(flag):

        q_n  = input("Enter a number    :: ")

        if(int(q_n) == 1):
            print("\nYou have selected an option to visualize the i/p data and"
                  + " convert the data to vectors")
            nb = NaiveBayes.NaiveBayes()
            data = nb.convert()


        elif(int(q_n) == 2):
            print("\nYou have selected an option for Naive Bayes classifier\n")
            nb = NaiveBayes.NaiveBayes()
            data = nb.convert(0)
            print("\nCalculating the Likelihood and Prior\n")
            likelihood,prior = nb.train(data)
            train_accuracy = nb.classify(data, likelihood, prior)
            train_accuracy = float("{:.2f}".format(train_accuracy))
            print("\nThe training error rate is ::", \
                  train_accuracy,"%\n")
            test_accuracy = nb.test(data, likelihood, prior)
            test_accuracy = float("{:.2f}".format(test_accuracy))
            print("\nThe testing error rate is ::", \
                  test_accuracy,"%\n")


        elif(int(q_n) == 3):
            print("\nYou have selected an option for KNN classifier\n")
            print("Please NOTE it will takes 15 minutes for KNN to run\n")
            knn = KNearestNeighbours.KNearestNeighbours()
            data = knn.convert(0)
            print("\nEvaluating the testing error in KNN using different k values\n")

            for k in range(1,11):
                testErrors.append(knn.classify(data, k))
                print("\nThe testing error rate for k = ",k,"is :",testErrors[-1],"\n")

            print("\nEvaluating the training error in KNN using different k values\n")

            for k in range(1,11):
                trainErrors.append(knn.train(data, k))
                print("\nThe training error rate for k = ",k,"is :",trainErrors[-1],"\n")

            print("\nEvaluating the testing error in KNN using Model Averaging\n")

            print("\nThe testing error for Model Averaging ","is :",knn.average(data),"\n")

        elif(int(q_n) == 4):
            print("You have selected an option to utilize PCA\n")
            libpca.funcPCA()

        else:
            print("You have selected a wrong choice, Please try again")

        flag = int(input("\nDo you want to continue, type 1 for yes, 0 otherwise :: "))



main()