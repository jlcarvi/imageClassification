# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:03:29 2019

@author: jlcar
"""
import cv2
import os
import matplotlib.pyplot as plt 
import random
import csv
import numpy as np
from numpy import save
from sklearn.model_selection import train_test_split

def generateTrainTest(datasetPath,trainPercentge,trainPathDataset,testPathDataset):
    classes=[]
    image_dataset=[]
  
    #define class number
    for r, d, f in os.walk(datasetPath):
        class_id=1
        for folder in d:
            print("Class: ",folder,class_id)
            classes.append([folder,class_id])
            class_id=class_id+1
   
        
    for class_name,class_id in classes:
        for image in os.listdir(datasetPath+class_name): 
            imagePath=datasetPath+class_name+"\\"+image #read original image
            image_dataset.append([imagePath,image,class_name,class_id])
            
    #Convert dataset to np array to make it easier shuffle and split wint sllearn
    np_image_dataset=np.array(image_dataset) 
  
    x_train, x_test, y_train, y_test = train_test_split(np_image_dataset[:,0:3],
                                                        np_image_dataset[:,3], test_size=0.3, random_state=0)

    #send training and testing images to train and test directory 
    for train_image,image_name,class_name in x_train:
        #print(train_image+"    -----> "+trainPathDataset+image_name)
        im=cv2.imread(train_image)
        cv2.imwrite(trainPathDataset+class_name+"\\"+image_name,im)
    
    for test_image,image_name,class_name in x_test:
        #print(test_image+"    -----> "+testPathDataset+image_name)
        im=cv2.imread(test_image)
        cv2.imwrite(testPathDataset+class_name+"\\"+image_name,im)
    
    #Generate csv files for training and testing datasets
    csvTrainLabels=np.concatenate((x_train[:,1:2], y_train.reshape(-1,1)), axis=1)
    csvTestLabels=np.concatenate((x_test[:,1:2], y_test.reshape(-1,1)), axis=1)
        
    print(csvTestLabels.shape)
    generateCSV(datasetPath+"testLabels.csv", csvTestLabels)
    generateCSV(datasetPath+"trainLabels.csv", csvTrainLabels)
    
    

def generateCSV(csvPath, data):
    with open(csvPath, 'w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    csvFile.close()


if __name__=='__main__':
    datasetPath="data\\datasets\\cards\\treatedData\\"
    trainPercentge=10
    trainPathDataset="data\\datasets\\cards\\train\\"
    testPathDataset="data\\datasets\\cards\\test\\"
    
    generateTrainTest(datasetPath,trainPercentge,trainPathDataset,testPathDataset)
    
   
    
    