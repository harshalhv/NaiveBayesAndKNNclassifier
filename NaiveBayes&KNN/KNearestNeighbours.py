# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
from collections import Counter

class KNearestNeighbours:
    class Data:
        def __init__(self):
            self.train = []
            self.test = []

    class Image:
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
            doc = self.Image()
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
            doc = self.Image()
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

    def distance(self, Image1, Image2):
        dis0 = np.sum(np.power(Image1.inp_data,Image2.inp_data,2))
        return dis0

    def getLabel(self, dis,trainLabels, k):
        if (len(dis)!=len(trainLabels)):
            print("This is pretty wrong")
        temp = dis.argsort()[:k][::-1]
        temp1 = [trainLabels[i] for i in temp]
        temp2 = Counter(temp1)
        return temp2.most_common(1)[0][0]

    def getLabelOverModels(self, dis,trainLabels):
        temp3 = []
        for k in range(1,11):
            if (len(dis)!=len(trainLabels)):
                print("This is pretty wrong")
            temp = dis.argsort()[:k][::-1]
            temp1 = [trainLabels[i] for i in temp]
            temp2 = Counter(temp1)
            temp3.append(temp2.most_common(1)[0][0])
        return Counter(temp3).most_common(1)[0][0]


    def classify(self, data,  k):
        trainMatrix = np.zeros([len(data.train),len(data.train[0].inp_data)])
        trainLabels = np.zeros(len(data.train))
        for i,image in enumerate(data.train):
            trainMatrix[i] = image.inp_data
            trainLabels[i] = image.num

        err = 0
        for image in data.test:
            dis = (np.sum(np.power((trainMatrix - image.inp_data),2),axis = 1))
            label = self.getLabel(dis,trainLabels, k)
            if(int(label)!=int(image.num)):
                err+=1
        return ((err/len(data.test)))*100


    def train(self, data, k):
        trainMatrix = np.zeros([len(data.train),len(data.train[0].inp_data)])
        trainLabels = np.zeros(len(data.train))
        for i,image in enumerate(data.train):
            trainMatrix[i] = image.inp_data
            trainLabels[i] = image.num

        err = 0
        for image in data.train:
            dis = (np.sum(np.power((trainMatrix - image.inp_data),2),axis = 1))
            label = self.getLabel(dis,trainLabels, k)
            if(int(label)!=int(image.num)):
                err+=1
        return ((err/len(data.train)))*100


    def average(self, data):
        trainMatrix = np.zeros([len(data.train),len(data.train[0].inp_data)])
        trainLabels = np.zeros(len(data.train))
        for i,image in enumerate(data.train):
            trainMatrix[i] = image.inp_data
            trainLabels[i] = image.num

        err = 0
        for image in data.test:
            dis = (np.sum(np.power((trainMatrix - image.inp_data),2),axis = 1))
            label = self.getLabelOverModels(dis,trainLabels)
            if(int(label)!=int(image.num)):
                err+=1
        return ((err/len(data.test)))*100