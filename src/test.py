from sklearn.datasets import fetch_olivetti_faces
import numpy as np

import matplotlib.pyplot as plt 
plt.style.use('dark_background') 
  
import os as os

import tensorflow as tf 
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dense, Dropout
  
  
from model import model_init

# Variable global
model_path = "./src/trained_model/siamese.h5"

def stiches(X, n_samples): # affichage d'un ensemble d'image
    x_dim, y_dim, *_ = X[0].shape
    return np.reshape(X.swapaxes(0,1), (x_dim, y_dim*n_samples)) 

def generate_test_image_pairs(images_dataset, labels_dataset, image):
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(label,
                                    [index for index, curr_label in enumerate(labels_dataset) if
                                        label == curr_label])
  
    pair_images = []
    pair_labels = []
    for label, indices_for_label in label_wise_indices.items():
        test_image = images_dataset[np.random.choice(indices_for_label)]
        pair_images.append((image, test_image))
        pair_labels.append(label)
    return np.array(pair_images), np.array(pair_labels)


if __name__ == "__main__":
    image = images_dataset[-1] # a random image as test image
    test_image_pairs, test_label_pairs = generate_test_image_pairs(images_dataset, labels_dataset, image) # produce an array of test image pairs and test label pairs

    # for each pair in the test image pair, predict the similarity between the images
    for index, pair in enumerate(test_image_pairs):
        pair_image1 = np.expand_dims(pair[0], axis=-1)
        pair_image1 = np.expand_dims(pair_image1, axis=0)
        pair_image2 = np.expand_dims(pair[1], axis=-1)
        pair_image2 = np.expand_dims(pair_image2, axis=0)
        prediction = model.predict([pair_image1, pair_image2])[0][0]