# Fashion minist 10 class DNN example
# code is change from https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb
from keras.datasets import fashion_mnist
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

#print(training_labels[0])
#print(training_images[0])

print(training_images.shape)
print(training_labels.shape)

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

history = model.fit(training_images, training_labels, epochs=10, verbose=2, validation_data=(test_images, test_labels))
print(model.summary())

result = model.predict(test_images)
print(result[9])
print(test_labels[9])

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label ='Validation acc')
plt.title("train and validation accuracy")
plt.show()


#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print(test_acc)
