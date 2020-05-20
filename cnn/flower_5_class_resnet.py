#https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Network_Keras.ipynb

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint

img_height,img_width = 224,224
num_classes = 6

base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)


sgd = optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.0001)
model.compile(optimizer= sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#model.fit(X_train, Y_train, epochs = 100, batch_size = 64)