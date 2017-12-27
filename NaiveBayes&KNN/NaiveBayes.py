# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 07:02:51 2017

@author: Yonarp
"""

import os
import random
import numpy as np

class NaiveBayes:
    class Data:
        def __init__(self):
            self.train = []
            self.test = []

    class Doc:
        def __init__(self):
            self.inp_data = 0
            self.num = 0
            self.width = 0
            self.height = 0

    def convert(self, display=1):
        if(display==1):
            self.visualize()
            print("\nConverting all the data to Vector started\n")
        data = self.Data()
        testDigits = os.listdir('./Data/testDigits/')
        trainingDigits = os.listdir('./Data/trainingDigits/')
        i=0
        for fileName in trainingDigits:
            doc = self.Doc()
            with open("./Data/trainingDigits/"+fileName) as file:
                doc.num = int(fileName.rstrip(".txt").split("_")[0])
                file_vec  = ""
                j=0
                for line in file:
                    file_vec +=line
                    j+=1
                self.height = j; self.width = len(file_vec)/j
                file_vec = np.array(list(map(int,file_vec.replace("\n",""))))
                doc.inp_data = file_vec
                data.train.append(doc)
            i+=1
            if(i%20==0 and display==1):
                print(".",end= '')
        print("")
        i=0
        for fileName in testDigits:
            doc = self.Doc()
            with open("./Data/testDigits/"+fileName) as file:
                doc.num = int(fileName.rstrip(".txt").split("_")[0])
                file_vec  = ""
                for line in file:
                    file_vec +=line
                file_vec = np.array(list(map(int,file_vec.replace("\n",""))))
                doc.inp_data = file_vec
                data.test.append(doc)
            i+=1
            if(i%20==0 and display ==1):
                print(".",end= '')
        if(display==1):
            print("\nConverting all the data to Vector DONE\n")
        return data


    def visualize(self):

        trainingDigits = os.listdir('./Data/trainingDigits/')

        print("\nDisplaying one Image for each number trainingDigits\n")
        numSet = set()
        fileName = random.choice(trainingDigits)
        for fileName in trainingDigits:
            doc = self.Doc()
            with open("./Data/trainingDigits/"+fileName) as file:
                doc.num = int(fileName.rstrip(".txt").split("_")[0])
                if doc.num not in numSet:
                    print("\nThe number to preinted now is :: ", doc.num, "\n")
                    for line in file:
                        print(line,end= '')
                    numSet.add(doc.num)



    def displayVector(self,data):
        print("\nDisplaying one random Image represented as Numpy Vector from trainingDigits\n")
        fileName = random.choice(data.train)
        print (fileName.data)
        print ("The corrosponding number for this vector is :: ", fileName.num, "\n")

        print("\nDisplaying one random Image represented as Numpy Vector from testDigits\n")
        fileName = random.choice(data.test)
        print (fileName.data)
        print ("The corrosponding number for this vector is :: ", fileName.num, "\n")

    def train(self, data):

        numSet = list(set([i.num for i in data.train]))
        numSet.sort()
        prior = []
        numDocs = dict()
        for i in numSet:
            prior.append(len([j.num for j in data.train  if j.num == i])/(len(data.train)))
            numDocs[i] = len([j.num for j in data.train  if j.num == i])
        numFeatures = np.zeros([len(numSet),len(data.train[0].inp_data)])
        likelihood = np.zeros([len(numSet),len(data.train[0].inp_data)])
        for image in data.train:
            numFeatures[image.num]+=image.inp_data
        for i in numSet:
            likelihood[i] = (numFeatures[i]+ 1 )/(numDocs[i]+2)
        return likelihood,np.array(prior)

    def classify(self, data, likelihood, prior):

        error = 0
        f = lambda x : 1 if x > 0 else 0
        vf = np.vectorize(f)
        for image in data.train:
            o_put = np.log(np.dot(likelihood,vf(image.inp_data.T)))+np.log(prior)
            o_put = np.log(np.dot((1-likelihood),(1-image.inp_data.T)))
            o_put = o_put.argmax()
            if(o_put!=image.num):
                error+=1
        return ((error/len(data.train)))*100


    def test(self, data, likelihood, prior):

        error = 0
        f = lambda x : 1 if x > 0 else 0
        vf = np.vectorize(f)
        for image in data.test:
            o_put = np.log(np.dot(likelihood,vf(image.inp_data.T)))+np.log(prior)
            o_put = np.log(np.dot((1-likelihood),(1-image.inp_data.T)))
            o_put = o_put.argmax()
            if(o_put!=image.num):
                error+=1
        return ((error/len(data.test)))*100






