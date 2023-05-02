from sklearn.datasets import fetch_olivetti_faces
import numpy as np

import matplotlib.pyplot as plt 
plt.style.use('dark_background') 
  
import os
import tensorflow as tf 
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dense, Dropout

from model import model_init
from test import predict_on_image, load_model
# Variable global
model_path = "./src/trained_model/siamese.h5"
 
def generate_train_image_pairs(images_dataset, labels_dataset): 
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(label,
                                      [index for index, curr_label in enumerate(labels_dataset) if
                                       label == curr_label])
    
    pair_images = []
    pair_labels = []
    for index, image in enumerate(images_dataset):
        pos_indices = label_wise_indices.get(labels_dataset[index])
        pos_image = images_dataset[np.random.choice(pos_indices)]
        pair_images.append((image, pos_image))
        pair_labels.append(1)

        neg_indices = np.where(labels_dataset != labels_dataset[index])
        neg_image = images_dataset[np.random.choice(neg_indices[0])]
        pair_images.append((image, neg_image))
        pair_labels.append(0)
        
    return np.array(pair_images), np.array(pair_labels) # pairs d'image et label (0 ou 1)
 

def save_model(model, loss):
    global model_path
    """Write logs and save the model"""
    #train_summary_writer = tf.summary.create_file_writer("./tmp/log")
    #with train_summary_writer.as_default():
    #    tf.summary.scalar("loss", loss)
    model.save(model_path) 
     
    
if __name__ == "__main__":   
    # Creation du model
    
    model = model_init()
    #model = load_model(model)
    
    olivetti = fetch_olivetti_faces()
    # On se reservera deux images pour le test
    x = olivetti.images[:100] # de forme (400-2, 64, 64), dtype=float32, valeurs entre 0.0 et
    y = olivetti.target[:100] # de forme (400-2,), dtype=int32, labels

    images_pair, labels_pair = generate_train_image_pairs(x, y)
    loss = model.fit([images_pair[:, 0], images_pair[:, 1]], labels_pair[:], validation_split=0.1, batch_size=64, epochs=1000)
    
    save_model(model, loss)
    
    # Prediction avec le model Siamois
    image = x[1] # a random image as test image
    predict_on_image(model, image, x, y)
       
