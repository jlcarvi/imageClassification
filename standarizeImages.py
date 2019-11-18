# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:44:00 2019

@author: jlcar
"""

import cv2
import os
import matplotlib.pyplot as plt 

#RENAME image files in each class folder
#INPUT:datasetPath: is the path where the classes folders are located             
#OUTPUT: renamed images in the same directory   
def image_rename(datasetPath):
    classes = []
    for r, d, f in os.walk(datasetPath):
        for folder in d:
            classes.append(folder)
    
    for c in classes:
        i=0        
        for image in os.listdir(datasetPath+"\\"+c): 
            dst =datasetPath+"\\"+c+"\\"+c+"-"+ str(i) + ".jpg"
            src =datasetPath+"\\"+c+"\\"+ image 
          
            os.rename(src, dst) 
            i += 1

#Resize images in class folders
#INPUT:
#   datasetPath: path where the folders with classes are located
#   height of new image, width of new image               
def image_standarize(datasetPath,height,width):
    classes = []
    for r, d, f in os.walk(datasetPath):
        for folder in d:
            classes.append(folder)
    
    for c in classes:
        print("***START RESIZING IMAGES IN CLASS "+c) 
        for image in os.listdir(datasetPath+"\\"+c): 
             im=cv2.imread(datasetPath+"\\"+c+'\\'+image) #read original image
             im=cv2.resize(im,(height,width)) 
             os.remove(datasetPath+"\\"+c+'\\'+image) #remove original image
             cv2.imwrite(datasetPath+"\\"+c+'\\'+image,im)
    print("Resized dataset:")   



######** TEST FUNCTION ********
#TO TEST CV2 funciotns
def image_reize(imagePath,height,width):
    im=cv2.imread(imagePath)
    heightOr, widthOr, depthOr = im.shape
    print(heightOr,widthOr, depthOr)
    
    ims=cv2.resize(im,(height,width))
    heights, widths, depths = ims.shape
    print(heights,widths, depths)
    
    print("Resize: ",imagePath)   
    #cv2.imshow("Resized image", im)
    #cv2.waitKey(0)
    Titles =["Original", "Resized"] 
    images =[im,ims] 
    os.remove(imagePath) 
    cv2.imwrite("treatedData\\testR.jpg",ims)
    for i in range(len(images)): 
        plt.subplot(1, 2, i + 1) 
        plt.title(Titles[i]) 
        plt.imshow(images[i]) 
        
    plt.show() 
    

    
if __name__=='__main__':
    height = 100
    width = 100
    raw_data = 'rawdata'
    data_path = 'data'
    datasetPath='treatedData'
   # image_rename(datasetPath)
    image_standarize(datasetPath,height,width)
    #image_reize("treatedData\\test.jpg",height,width)
    
    