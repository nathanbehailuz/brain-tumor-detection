import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

image_directory = "./dataset" 

dataset = []
label = []

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
                image = image.resize((64, 64))
                dataset.append(np.array(image))
                label.append(0 if label_value == 'no' else 1)

    process_images(no_tumor_images, 'no')
    process_images(yes_tumor_images, 'yes')

    return np.array(dataset), np.array(label)

dataset, label = load_and_process_images(image_directory)

# divide the dataset into train test and split 
# 80% train, 20% test 

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2 , random_state = 0)

# Reshape = (n, image_width, image_height, n_channel)