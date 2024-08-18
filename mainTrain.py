import cv2
import os
from PIL import Image
import numpy as np

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

    return dataset, label

dataset, label = load_and_process_images(image_directory)

print(len(label))

