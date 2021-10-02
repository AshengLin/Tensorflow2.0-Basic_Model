import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255
# print(x_train.shape)  # (60000, 28, 28)
x_train = x_train.reshape((60000, 28, 28, 1))  # 灰階

from tensorflow import keras
from tensorflow.keras import layers
import os

cnn_model = keras.Sequential(name='CNN')
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(128, activation='relu'))
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(layers.Dense(10, activation='softmax'))  # output


cnn_model.compile(optimizer='Adam',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
cnn_model.fit(x_train, y_train, epochs=10)

if os.path.exists('model.png'):
    os.remove('model.png')
keras.utils.plot_model(cnn_model, show_shapes='True', to_file='model.png')
