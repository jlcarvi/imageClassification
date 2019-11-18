# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:23:46 2019

@author: jlcar
"""
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import sys

PATH="data\\datasets\\cards\\"
batch_size = 128
epochs = 10
IMG_HEIGHT = 100
IMG_WIDTH = 100

# Define a simple sequential model
def create_model():
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

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model



#Function to load previously trainind weights in a empty model
#please check for further details:
#    https://www.tensorflow.org/tutorials/keras/save_and_load
def load_model_from_chp():
    test_dir = os.path.join(PATH, 'test')
    # Create a basic model instance
    model = create_model()

    # Display the model's architecture
    model.summary()
    
    #Test the empty model   
    test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='sparse') #class_mode='binary'

    loss, acc = model.evaluate(test_data_gen, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
    
    
    #******* Loads the weights
    checkpoint_path = "card_models/training_chp/cp.ckpt"
    model.load_weights(checkpoint_path)

    #******** Re-evaluate the model
    loss,acc = model.evaluate(test_data_gen, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

def manually_load_weight():
    test_dir = os.path.join(PATH, 'test')
    # Create a basic model instance
    model = create_model()

    # Display the model's architecture
    model.summary()
    
    #Test the empty model   
    test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='sparse') #class_mode='binary'

    loss, acc = model.evaluate(test_data_gen, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
    
    
    #******* Loads the weights
    checkpoint_path = "card_models/manuallySavedWeights/manualSaved'"
    model.load_weights(checkpoint_path)

    #******** Re-evaluate the model
    loss,acc = model.evaluate(test_data_gen, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


def load_model():
    test_dir = os.path.join(PATH, 'test')
    new_model = tf.keras.models.load_model('card_models/entireModel')
    

    # Check its architecture
    new_model.summary()
    # Evaluate the restored model
    #Test the empty model   
    test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='sparse') #class_mode='binary'

    
    class_dictionary = test_data_gen.class_indices
    print(class_dictionary)
    loss, acc = new_model.evaluate(test_data_gen, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
    return new_model, class_dictionary

def predict( model, class_dictionary):
    #try:
        img = cv2.imread('test.jpg')
        plt.imshow(img) 
        img = cv2.resize(img,(100,100))
        img=img/255.0
        img = np.reshape(img,[1,100,100,3])
        
        classes = model.predict_classes(img)
        print(classes)
    #except:
     #   print("Oops!",sys.exc_info()[0],"occured.")
if __name__=='__main__':
    #print(tf.__version__)
    #load_model_from_chp()
    model=load_model()
    predict(model)
    