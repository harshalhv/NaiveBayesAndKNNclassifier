# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:31:13 2017

@author: Yonarp
"""

import NaiveBayes
from sklearn.decomposition import PCA
import numpy as np
import KNearestNeighbours

def funcPCA():
    nb = NaiveBayes.NaiveBayes()
    data = nb.convert(0)
    pca = PCA(n_components=512)
    print("\nNaive Byes after PCA to reduced data dimension to 512\n")
    featureMatrix = np.zeros([len(data.train),1024])
    for i, image in enumerate(data.train):
        featureMatrix[i]=image.inp_data
    featureMatrix = pca.fit(featureMatrix).transform(featureMatrix)
    for i, image in enumerate(data.train):
        data.train[i].inp_data = featureMatrix[i]
    featureMatrix = np.zeros([len(data.test),1024])
    for i, image in enumerate(data.test):
        featureMatrix[i]=image.inp_data

    featureMatrix = pca.fit(featureMatrix).transform(featureMatrix)

    for i, image in enumerate(data.test):
        data.test[i].inp_data = featureMatrix[i]
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


    print("\nKNN after PCA to reduced data dimension to 512\n")




    print("Please NOTE it will takes 15 minutes for KNN to run\n")
    knn = KNearestNeighbours.KNearestNeighbours()
    data = knn.convert(0)
    print("\nEvaluating the testing error in KNN using different k values\n")
    testErrors = []
    trainErrors = []
    for k in range(1,11):
        testErrors.append(knn.classify(data, k))
        print("\nThe testing error rate for k = ",k,"is :",testErrors[-1],"\n")

    print("\nEvaluating the training error in KNN using different k values\n")

    for k in range(1,11):
        trainErrors.append(knn.train(data, k))
        print("\nThe training error rate for k = ",k,"is :",trainErrors[-1],"\n")



