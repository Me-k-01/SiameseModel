import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dense, Dropout

import matplotlib.pyplot as plt 
from tensorflow.keras import backend as k
plt.style.use('dark_background') 
 
def create_model():
    inputs = Input((64, 64, 1))
    x = Conv2D(96, (11, 11), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(384, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    pooledOutput = Dense(1024)(pooledOutput)
    outputs = Dense(128)(pooledOutput)

    model = tf.keras.Model(inputs, outputs)
    return model

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))


def model_init():
    
    # Creation du model
    feature_extractor = create_model()
    imgA = Input(shape=(64, 64, 1))
    imgB = Input(shape=(64, 64, 1))
    featA = feature_extractor(imgA)
    featB = feature_extractor(imgB)
    
    distance = Lambda(euclidean_distance)([featA, featB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = tf.keras.Model(inputs=[imgA, imgB], outputs=outputs)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model