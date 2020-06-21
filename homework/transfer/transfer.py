from __future__ import print_function
import numpy as np
import datetime
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend

now = datetime.datetime.now

img_rows, img_cols = 28, 28
data = mnist.load_data()
fit_params = {
        'batch_size': 128,
        'epochs': 5
        }


def train_model(model, data, num_classes, fit_params):
    input_shape = get_input_shape()
    X_train, X_test = modify_feature_shapes(data, input_shape)
    return X_train
   # y_train, y_test = modify_target_to_categorical(data, num_classes)
    #current_time = now()
    #model.fit(X_train, y_train, verbose=1, validation_data=(X_test, y_test), **fit_params)
    #print('Training time: %s' % (now() - t))
    #score = model.evaluate(X_test, y_test, verbose=0)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])


def get_input_shape():
    if backend.image_data_format() == 'channels_first':
        return (1, img_rows, img_cols)
    else:
        return (img_rows, img_cols, 1)


def modify_feature_shapes(data, input_shape):
    feature_data = np.array(collection[0] for collection in data)
    reshape = lambda features, input_shape: features.reshape((features.shape[0],) + input_shape) 
    return (reshape(feature_data, input_shape) for features in feature_data)


def modify_target_to_categorical(data, num_classes):
    targets = (collection[1] for collection in data)
    return (to_categorical(target, num_classes) for target in targets)


feature_layers = [
        Conv2D(filters=32, kernel_size=3,
            padding='valid',
            input_shape=get_input_shape()),
        Activation('relu'),
        Conv2D(filters=32, kernel_size=3),
        Activation('relu'),
        MaxPooling2D(pool_size=2),
        Dropout(0.25),
        Flatten()
        ]

classification_layers = [
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(5),
        Activation('softmax')
        ]

model = Sequential(feature_layers + classification_layers)

print(train_model(model, data, 5, fit_params))

