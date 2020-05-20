# this part is implemented by Keras
from keras.datasets import fashion_mnist
from keras import models
from keras import layers
#import matplotlib.pyplot as plt

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

#print(training_labels[0])
#print(training_images[0])

#print(training_images.shape)
#print(training_labels.shape)

#plt.imshow(training_images[10002])
#plt.show()


training_images = training_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)
print(model.summary())

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

'''
# this part is implemented by tensorflow
# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
'''

'''
# simple predict
import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
inputs = Input(shape=(3,2,4))

# Define a model consisting only of the Flatten operation
prediction = Flatten()(inputs)
model = Model(inputs=inputs, outputs=prediction)

X = np.arange(0,24).reshape(1,3,2,4)
print(X)
#[[[[ 0  1  2  3]
#   [ 4  5  6  7]]
#
#  [[ 8  9 10 11]
#   [12 13 14 15]]
#
#  [[16 17 18 19]
#   [20 21 22 23]]]]
model.predict(X)
#array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
#         11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
#         22.,  23.]], dtype=float32)
'''
