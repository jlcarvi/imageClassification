# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:49:12 2019
CARDS CLASSIFICATION
@author: jlcar
"""
from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    #if(logs.get('loss')<0.4):
    if(logs.get('accuracy')>0.91):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True



PATH="data\\datasets\\cards\\"
batch_size = 128
epochs = 10
IMG_HEIGHT = 100
IMG_WIDTH = 100

def cards_classify():
    callbacks = myCallback()
    train_dir = os.path.join(PATH, 'train')
    test_dir = os.path.join(PATH, 'test')

    total_train = totalFiles(train_dir)
    total_test=totalFiles(test_dir)
    print("Total training: ",total_train)
    print("Total testing: ",total_test)


    #Data preparation
    #1. Read images from the disk.
    #2. Decode contents of these images and convert it into proper grid format as per their RGB content.
    #3. Convert them into floating point tensors.
    #4. Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.
    train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
    test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='sparse') #class_mode='binary'

    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='sparse') #class_mode='binary'


    #class_mode='sparse' : generate float numbers as labels eg 1., 2., 3. ,...
    #class_mode='categorical' : generate one hot bit
    

    #*********************Augmentation with TF
    image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                               class_mode='sparse')
    
    
    augmented_images = [train_data_gen[0][0][0] for i in range(4)]
    plotImages(augmented_images)
    
    
    
    #### VISUALIZE IMAGES####
    #he return value of next function is in form of (x_train, y_train) 
    #where x_train is training features and y_train, its labels.
    sample_training_images, sample_label = next(train_data_gen)
   # plotImages(sample_training_images[:4])    
    
    
    ###### CREATE THE MODEL WITHOUT DROPOUT###########
    '''
    model = tf.keras.models.Sequential([
     tf.keras.layers.Conv2D(16, (3,3), padding='same', activation=tf.nn.relu, input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512, activation=tf.nn.relu),
     tf.keras.layers.Dense(4, activation=tf.nn.softmax) ])
    '''
    
    ###### CREATE THE MODEL WITH DROPOUT TO HANDLE OVERFITING###########
    model = tf.keras.models.Sequential([
     tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512, activation='relu'),
     tf.keras.layers.Dense(4, activation=tf.nn.softmax)])

    
    ## COMPILE THE MODEL
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    
    
    
    #model.fit(train_data_gen, epochs=10, steps_per_epoch=30)

    model.summary()
    
    #************** Train the model

    history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=test_data_gen,
            validation_steps=total_test // batch_size,
            callbacks=[callbacks])
    
    
    #****************** Visualize results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range =range(len(val_acc))
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    
    
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
    
    
    
# This function will plot images in the form of a grid with 1 row and 5 columns
# where images are placed in each column.
def plotImages(images_arr):
    
    fig, axes = plt.subplots(2, 2, figsize=(5,5))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    '''
    for i in range(4): 
        plt.subplot(1, 2, i + 1) 
        #plt.title(Titles[i]) 
        plt.imshow(images_arr[i]) 
        
    plt.show() 
    '''

#count the total number of files without directories in path
def totalFiles(path):
    classes=[]
    for r, d, f in os.walk(path):
        for folder in d:
            classes.append(folder)
    
    count=0
    for c in classes:
        for image in os.listdir(path+"\\"+c): 
            count += 1
    
    return count        

if __name__=='__main__':
    #print(tf.__version__)
    cards_classify()
    
    