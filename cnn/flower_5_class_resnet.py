#https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Network_Keras.ipynb

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def resNet50_model():
    base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (224,224,3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(5, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)

    sgd = optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True)

    model.compile(optimizer= sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == "__main__":
    trdata = ImageDataGenerator()
    traindata = trdata.flow_from_directory(directory="C:\\Work\\data\\flower_photos_small\\train",
                                           target_size=(224, 224))
    tsdata = ImageDataGenerator()
    testdata = tsdata.flow_from_directory(directory="C:\\Work\\data\\flower_photos_small\\validation",
                                          target_size=(224, 224))

    checkpoint = ModelCheckpoint("resnet_50.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

    model = resNet50_model()

    hist = model.fit_generator(generator=traindata, validation_data=testdata, steps_per_epoch=100, validation_steps=10,
                               epochs=10, callbacks=[checkpoint,early])

    plt.plot(hist.history["acc"])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show()