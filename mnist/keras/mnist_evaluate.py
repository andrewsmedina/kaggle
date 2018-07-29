from keras.models import load_model
from keras.datasets import mnist
import keras
import pandas as pd
import numpy as np

batch_size = 128
num_classes = 10

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

y_test = keras.utils.to_categorical(y_test, num_classes)

model = load_model("my_model.h5")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
