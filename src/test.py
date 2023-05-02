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

def load_model(model):
    global model_path
    if not os.path.isfile(model_path):
        return RuntimeError("Error loading model", model_path)

    print("Loading model from", model_path)
    model.load_weights(model_path)
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

def predict_on_image(model, image, x, y): 
    test_image_pairs, _ = generate_test_image_pairs(x, y, image) # produce an array of test image pairs and test label pairs
 

    images = []
    preds = []

    # for each pair in the test image pair, predict the similarity between the images
    for index, pair in enumerate(test_image_pairs):
        pair_image1 = np.expand_dims(pair[0], axis=-1)
        pair_image1 = np.expand_dims(pair_image1, axis=0)
        pair_image2 = np.expand_dims(pair[1], axis=-1)
        pair_image2 = np.expand_dims(pair_image2, axis=0)
        #pair_img = np.array([pair_image1, pair_image2])
        prediction = model.predict([pair_image1, pair_image2])[0][0]
        preds.append(prediction)
        images.append(stiches(np.array([pair[0], pair[1]]), 2))

        #axs[index].imshow(stiches(np.array([pair[0], pair[1]]), 2), cmap='gray') 
        #axs[index].set_title(str(prediction))
        #axs[index].update(wspace=0.5, hspace=0.5)
        #axs[index].tight_layout()
    #images = [img for _, img in sorted(zip(preds, images))]
    #preds = sorted(preds) 

    fig, axs = plt.subplots(len(test_image_pairs)) 
    fig.subplots_adjust(hspace=1, wspace=1)
    for i in range(len(images)):
        axs[index].imshow(images[i], cmap='gray') 
        axs[index].set_title(str(preds[i]))
    #plt.savefig('./pred_by_patch_' + str(FLAGS.patch_size_in) +  '_to_' + str(FLAGS.patch_size_out) + '.png')
    #plt.setp([a.get_xticklabels() for a in axs ], visible=False)
    #plt.setp([a.get_yticklabels() for a in axs ], visible=False)

    fig.tight_layout() 
    plt.show()


if __name__ == "__main__":
    
    model = model_init()
    model = load_model(model)

    olivetti = fetch_olivetti_faces()
    # On se reservera deux images pour le test
    x = olivetti.images[:100] # de forme (400-2, 64, 64), dtype=float32, valeurs entre 0.0 et
    y = olivetti.target[:100] # de forme (400-2,), dtype=int32, labels

    # Prediction avec le model Siamois
    image = x[-1] # a random image as test image
    predict_on_image(model, image, x, y)
