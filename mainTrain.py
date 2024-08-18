import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils  import normalize 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense 

image_directory = "./dataset" 

dataset = []
label = []
INPUT_SIZE = 64

def load_and_process_images(image_directory):
    no_tumor_images = os.listdir(image_directory + '/no')
    yes_tumor_images = os.listdir(image_directory + '/yes')

    dataset = []
    label = []

    def process_images(image_list, label_value):
        for image_name in image_list:
            if image_name.split('.')[1] == 'jpg':
                image = cv2.imread(image_directory + f'/{label_value}/{image_name}')
                image = Image.fromarray(image, 'RGB')
                image = image.resize((INPUT_SIZE, INPUT_SIZE))
                dataset.append(np.array(image))
                label.append(0 if label_value == 'no' else 1)

    process_images(no_tumor_images, 'no')
    process_images(yes_tumor_images, 'yes')

    return np.array(dataset), np.array(label)

dataset, label = load_and_process_images(image_directory)

# divide the dataset into train test and split 
# 80% train, 20% test 

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2 , random_state = 0)

# normalize the data
x_train = normalize(x_train, axis = 1)
x_test = normalize(x_test, axis = 1)

# model building 
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, 
          batch_size = 16, 
          verbose = 1, 
          epochs = 10, 
          validation_data = (x_test, y_test),
          shuffle = False)

model.save('BrainTumor10Epochs.h5')