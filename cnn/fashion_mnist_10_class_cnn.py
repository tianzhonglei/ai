# Fashion minist 10 class CNN example
# code is change from https://codelabs.developers.google.com/codelabs/tensorflow-lab3-convolutions/#0

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(training_images, training_labels, epochs=10, verbose=2, validation_data=(test_images, test_labels))

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label ='Validation acc')
plt.title("train and validation accuracy")
plt.show()

#test_loss, test_accuracy = model.evaluate(test_images, test_labels)
#print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy * 100))

#Test loss: 0.26751336119174957, Test accuracy: 89.95
#Test loss: 0.2722022842407227, Test accuracy: 90.13

