import tensorflow as tf
from keras import models
from keras import layers
from keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = models.Sequential([
    layers.Flatten.input_shape(28, 28),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer, loss="sparse_categorical_crossentropy")

model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evalutate(test_images, test_labels)

print(test_acc)

#https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb