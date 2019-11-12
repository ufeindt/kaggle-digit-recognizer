import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers, callbacks
from keras.utils.np_utils import to_categorical

data_train = pd.read_csv('train.csv')

X =  np.array(data_train.drop(labels=['label'], axis=1)).reshape(-1, 28, 28, 1)
X = X/255

y = to_categorical(data_train['label'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# CNN Architecture
cnn = models.Sequential()
cnn.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                           input_shape=(28, 28, 1)))
cnn.add(layers.Activation('relu'))
cnn.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1)))
cnn.add(layers.Activation('relu'))
cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))

cnn.add(layers.Dropout(rate=0.25))

cnn.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)))
cnn.add(layers.Activation('relu'))
cnn.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)))
cnn.add(layers.Activation('relu'))
cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))

cnn.add(layers.Dropout(rate=0.25))

cnn.add(layers.Flatten())

cnn.add(layers.Dropout(rate=0.4))

cnn.add(layers.Dense(units=128))
cnn.add(layers.Activation('relu'))

cnn.add(layers.Dense(units=10))
cnn.add(layers.Activation('sigmoid'))
print(cnn.summary())

# sys.exit()

compile = cnn.compile(optimizer='adam',#optimizers.SGD(learning_rate=0.2),
                      loss='categorical_crossentropy', metrics=['accuracy'])

# Settting Callbacks
check_p = callbacks.ModelCheckpoint(filepath='mnist_cnn_{val_accuracy:.4f}.h5',
                                    monitor='val_accuracy', verbose=1,
                                    save_best_only=True, save_weights_only=False)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.95,
                                        patience=3, verbose=1, cooldown=2)
callb_l = [check_p, reduce_lr]

# Training Options
fit = cnn.fit(X_train, y_train, validation_data=(X_val, y_val),
              steps_per_epoch=100, epochs=50, verbose=1, callbacks=callb_l,
              validation_steps=10)

# Saving Model
cnn.save(filepath=r'mnist_cnn.h5', overwrite=True)
