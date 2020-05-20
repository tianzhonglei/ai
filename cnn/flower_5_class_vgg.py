#data : https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
#https://juejin.im/post/5d8b27396fb9a04dd85909ab
#https://www.jianshu.com/p/130171318888

import keras
from keras import optimizers
from keras.applications import VGG16
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def vgg_model():
    base_model = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))  # 不包含最后一层
    model = Flatten(name='Flatten')(base_model.output)
    model = Dense(5, activation='softmax')(model)  # 最后一层自定义
    model = Model(inputs=base_model.input, outputs=model, name='vgg16')
    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)  # SGD is better than Adam
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def vgg_model_self_1():
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=5, activation="softmax"))
    optimizer = optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model

def vgg_model_self_2():
    from keras.layers import Dropout, Flatten, Dense, Input
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    input_1 = Input((224, 224, 3))  # 输入224*224*3
    # 第一部分
    # 卷积 64深度，大小是3*3 步长为1 使用零填充 激活函数relu
    # 2次卷积 一次池化 池化尺寸2*2 步长2*2
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(input_1)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 64 224*224
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 64 112*112

    # 第二部分 2次卷积 一次池化
    # 卷积 128深度 大小是3*3 步长1 零填充
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 128 112*112
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 128 56*56

    # 第三部分 3次卷积 一次池化 卷积256 3*3
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 256 56*56
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 256 28*28

    # 第四部分 3次卷积 一次池化 卷积 512 3*3
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 512 28*28
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 512 14*14

    # 第五部分 3次卷积 一次池化 卷积 512 3*3
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 512 14*14
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 512 7*7

    x = Flatten()(x)  # 扁平化，用在全连接过渡
    # 第六部分 三个全连接
    # 第一个和第二个全连接相同 输出4096 激活relu 使用dropout，随机丢弃一半
    x = Dense(4096, activation="relu")(x)
    Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)
    Dropout(0.5)(x)  # 输出 4096 1*1
    # 第三个全连接层 输出 softmax分类
    out_ = Dense(5, activation="softmax")(x)
    model = Model(inputs=input_1, outputs=out_)
    sgd = optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    return model

if __name__ == "__main__":
    trdata = ImageDataGenerator()
    traindata = trdata.flow_from_directory(directory="C:\\Work\\data\\flower_photos_small\\train",
                                           target_size=(224, 224))
    tsdata = ImageDataGenerator()
    testdata = tsdata.flow_from_directory(directory="C:\\Work\\data\\flower_photos_small\\validation",
                                          target_size=(224, 224))

    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

    model = vgg_model_self_2()

    hist = model.fit_generator(generator=traindata, validation_data=testdata, steps_per_epoch=100, validation_steps=10,
                               epochs=10,callbacks=[checkpoint,early])

    plt.plot(hist.history["acc"])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show()


'''
  vgg_model_self_1 
  1/100 [..............................] - ETA: 2:26:44 - loss: 1.6782 - acc: 0.1875
  2/100 [..............................] - ETA: 2:38:24 - loss: 1.6919 - acc: 0.1719
  3/100 [..............................] - ETA: 2:31:10 - loss: 1.6854 - acc: 0.1771
  4/100 [>.............................] - ETA: 2:22:54 - loss: 1.6688 - acc: 0.1875
  5/100 [>.............................] - ETA: 2:16:52 - loss: 1.6550 - acc: 0.1812
  6/100 [>.............................] - ETA: 2:10:52 - loss: 1.6437 - acc: 0.1927
  7/100 [=>............................] - ETA: 2:06:00 - loss: 1.6386 - acc: 0.2009
  8/100 [=>............................] - ETA: 2:03:23 - loss: 1.6254 - acc: 0.2148
  9/100 [=>............................] - ETA: 2:02:06 - loss: 1.6238 - acc: 0.2188
 10/100 [==>...........................] - ETA: 1:59:44 - loss: 1.6199 - acc: 0.2313
 11/100 [==>...........................] - ETA: 1:57:04 - loss: 1.6234 - acc: 0.2216
 12/100 [==>...........................] - ETA: 1:54:37 - loss: 1.6169 - acc: 0.2344
 13/100 [==>...........................] - ETA: 1:52:17 - loss: 1.6127 - acc: 0.2332
 14/100 [===>..........................] - ETA: 1:49:49 - loss: 1.6084 - acc: 0.2344
 15/100 [===>..........................] - ETA: 1:47:53 - loss: 1.6036 - acc: 0.2375
 16/100 [===>..........................] - ETA: 1:46:22 - loss: 1.5931 - acc: 0.2500
 17/100 [====>.........................] - ETA: 1:45:01 - loss: 1.5928 - acc: 0.2482
 18/100 [====>.........................] - ETA: 1:44:15 - loss: 1.5946 - acc: 0.2448
 19/100 [====>.........................] - ETA: 1:43:10 - loss: 1.5849 - acc: 0.2615
 20/100 [=====>........................] - ETA: 1:41:21 - loss: 1.5839 - acc: 0.2562
 21/100 [=====>........................] - ETA: 1:39:38 - loss: 1.5784 - acc: 0.2604
 22/100 [=====>........................] - ETA: 1:38:30 - loss: 1.5756 - acc: 0.2599
 23/100 [=====>........................] - ETA: 1:37:20 - loss: 1.5731 - acc: 0.2595
 24/100 [======>.......................] - ETA: 1:36:01 - loss: 1.5720 - acc: 0.2591
 25/100 [======>.......................] - ETA: 1:34:43 - loss: 1.5758 - acc: 0.2587
 26/100 [======>.......................] - ETA: 1:34:24 - loss: 1.5729 - acc: 0.2668
 27/100 [=======>......................] - ETA: 1:34:29 - loss: 1.5710 - acc: 0.2627
 28/100 [=======>......................] - ETA: 1:34:30 - loss: 1.5680 - acc: 0.2645
 29/100 [=======>......................] - ETA: 1:34:18 - loss: 1.5626 - acc: 0.2705
 30/100 [========>.....................] - ETA: 1:33:18 - loss: 1.5613 - acc: 0.2708
 31/100 [========>.....................] - ETA: 1:32:40 - loss: 1.5577 - acc: 0.2722
 32/100 [========>.....................] - ETA: 1:30:45 - loss: 1.5507 - acc: 0.2774
 33/100 [========>.....................] - ETA: 1:29:46 - loss: 1.5490 - acc: 0.2794
 34/100 [=========>....................] - ETA: 1:28:50 - loss: 1.5468 - acc: 0.2813
 35/100 [=========>....................] - ETA: 1:27:51 - loss: 1.5411 - acc: 0.2867
 36/100 [=========>....................] - ETA: 1:26:45 - loss: 1.5351 - acc: 0.2917
 37/100 [==========>...................] - ETA: 1:25:17 - loss: 1.5280 - acc: 0.2991
 38/100 [==========>...................] - ETA: 1:23:28 - loss: 1.5186 - acc: 0.3060
 39/100 [==========>...................] - ETA: 1:21:53 - loss: 1.5144 - acc: 0.3094
 40/100 [===========>..................] - ETA: 1:20:17 - loss: 1.5053 - acc: 0.3149
 41/100 [===========>..................] - ETA: 1:18:43 - loss: 1.5054 - acc: 0.3209
 42/100 [===========>..................] - ETA: 1:17:15 - loss: 1.5028 - acc: 0.3222
 43/100 [===========>..................] - ETA: 1:16:08 - loss: 1.4940 - acc: 0.3256
 44/100 [============>.................] - ETA: 1:15:30 - loss: 1.4903 - acc: 0.3296
 45/100 [============>.................] - ETA: 1:14:41 - loss: 1.5204 - acc: 0.3251
 46/100 [============>.................] - ETA: 1:13:35 - loss: 1.5421 - acc: 0.3221
 47/100 [=============>................] - ETA: 1:12:28 - loss: 1.5404 - acc: 0.3219
 48/100 [=============>................] - ETA: 1:11:15 - loss: 1.5400 - acc: 0.3223
 49/100 [=============>................] - ETA: 1:09:55 - loss: 1.5393 - acc: 0.3234
 50/100 [==============>...............] - ETA: 1:08:34 - loss: 1.5403 - acc: 0.3225
 51/100 [==============>...............] - ETA: 1:07:14 - loss: 1.5398 - acc: 0.3217
 52/100 [==============>...............] - ETA: 1:05:55 - loss: 1.5388 - acc: 0.3210
 53/100 [==============>...............] - ETA: 1:04:24 - loss: 1.5386 - acc: 0.3208
 54/100 [===============>..............] - ETA: 1:03:03 - loss: 1.5401 - acc: 0.3201
 55/100 [===============>..............] - ETA: 1:01:52 - loss: 1.5418 - acc: 0.3188
 56/100 [===============>..............] - ETA: 1:00:42 - loss: 1.5415 - acc: 0.3204
 57/100 [================>.............] - ETA: 59:27 - loss: 1.5434 - acc: 0.3197  
 58/100 [================>.............] - ETA: 58:10 - loss: 1.5451 - acc: 0.3163
 59/100 [================>.............] - ETA: 56:59 - loss: 1.5436 - acc: 0.3168
 60/100 [=================>............] - ETA: 55:37 - loss: 1.5422 - acc: 0.3188
 61/100 [=================>............] - ETA: 54:11 - loss: 1.5414 - acc: 0.3202
 62/100 [=================>............] - ETA: 52:40 - loss: 1.5413 - acc: 0.3201
 63/100 [=================>............] - ETA: 51:18 - loss: 1.5385 - acc: 0.3230
 64/100 [==================>...........] - ETA: 49:42 - loss: 1.5371 - acc: 0.3235
 65/100 [==================>...........] - ETA: 48:18 - loss: 1.5320 - acc: 0.3267
 66/100 [==================>...........] - ETA: 46:59 - loss: 1.5346 - acc: 0.3260
 67/100 [===================>..........] - ETA: 45:43 - loss: 1.5391 - acc: 0.3254
 68/100 [===================>..........] - ETA: 44:23 - loss: 1.5419 - acc: 0.3215
 69/100 [===================>..........] - ETA: 43:00 - loss: 1.5412 - acc: 0.3223
 70/100 [====================>.........] - ETA: 41:40 - loss: 1.5420 - acc: 0.3195
 71/100 [====================>.........] - ETA: 40:13 - loss: 1.5417 - acc: 0.3198
 72/100 [====================>.........] - ETA: 38:49 - loss: 1.5418 - acc: 0.3197
 73/100 [====================>.........] - ETA: 37:21 - loss: 1.5416 - acc: 0.3200
 74/100 [=====================>........] - ETA: 35:55 - loss: 1.5411 - acc: 0.3191
 75/100 [=====================>........] - ETA: 34:30 - loss: 1.5405 - acc: 0.3198
 76/100 [=====================>........] - ETA: 33:03 - loss: 1.5404 - acc: 0.3185
 77/100 [======================>.......] - ETA: 31:37 - loss: 1.5397 - acc: 0.3196
 78/100 [======================>.......] - ETA: 30:12 - loss: 1.5385 - acc: 0.3196
 79/100 [======================>.......] - ETA: 28:50 - loss: 1.5374 - acc: 0.3210
 80/100 [=======================>......] - ETA: 27:26 - loss: 1.5356 - acc: 0.3221
 81/100 [=======================>......] - ETA: 26:02 - loss: 1.5327 - acc: 0.3251
 82/100 [=======================>......] - ETA: 24:37 - loss: 1.5317 - acc: 0.3257
 83/100 [=======================>......] - ETA: 23:12 - loss: 1.5312 - acc: 0.3255
 84/100 [========================>.....] - ETA: 21:51 - loss: 1.5297 - acc: 0.3269
 85/100 [========================>.....] - ETA: 20:30 - loss: 1.5297 - acc: 0.3252
 86/100 [========================>.....] - ETA: 19:08 - loss: 1.5281 - acc: 0.3273
 87/100 [=========================>....] - ETA: 17:46 - loss: 1.5279 - acc: 0.3282
 88/100 [=========================>....] - ETA: 16:22 - loss: 1.5241 - acc: 0.3305
 89/100 [=========================>....] - ETA: 14:59 - loss: 1.5238 - acc: 0.3299
 90/100 [==========================>...] - ETA: 13:37 - loss: 1.5207 - acc: 0.3315
 91/100 [==========================>...] - ETA: 12:14 - loss: 1.5191 - acc: 0.3326
 92/100 [==========================>...] - ETA: 10:51 - loss: 1.5218 - acc: 0.3321
 93/100 [==========================>...] - ETA: 9:29 - loss: 1.5202 - acc: 0.3332 
 94/100 [===========================>..] - ETA: 8:07 - loss: 1.5186 - acc: 0.3340
 95/100 [===========================>..] - ETA: 6:45 - loss: 1.5161 - acc: 0.3361
 96/100 [===========================>..] - ETA: 5:23 - loss: 1.5143 - acc: 0.3367
 97/100 [============================>.] - ETA: 4:02 - loss: 1.5129 - acc: 0.3374
 98/100 [============================>.] - ETA: 2:41 - loss: 1.5108 - acc: 0.3381
 99/100 [============================>.] - ETA: 1:20 - loss: 1.5096 - acc: 0.3385
100/100 [==============================] - 8239s 82s/step - loss: 1.5081 - acc: 0.3382 - val_loss: 1.3994 - val_acc: 0.4188

Epoch 00001: val_acc improved from -inf to 0.41875, saving model to vgg16_1.h5

'''


'''

import os

import cv2
import numpy as np

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import TensorBoard
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers




class vgg():
    def __init__(self, shape, num_classes, data_path, label_path, model_path):
        self.shape = shape
        self.num_classes = num_classes
        self.data_path = data_path
        self.label_path = label_path
        self.model_path = model_path
        self.log_path = "./logs"
        self.classes = self.classname()

    def classname(self, prepath="C://Work//data//flower_photos//"):
        # 数据集的类别序号和对应名称，注意的是序号是从1开始，而label中编码实际是从0开始
        classes = os.listdir(prepath)  # 类别序号和名称
        class_dict = {int(Class.split(".")[0]): Class.split(".")[1] for Class in classes[0:5]}
        return class_dict

    def generate_data(self, prepath="C://Work//data//flower_photos//"):
        classes = os.listdir(prepath)  # 类别序号和名称

        data_path = self.data_path
        label_path = self.label_path
        datas = []
        labels = []
        for i, abspath in enumerate(classes):  # prepath的每一个文件目录
            print(classes)
            img_names = os.listdir(prepath + abspath)
            for img_name in img_names:  # 子目录中的每一张图片
                img = cv2.imread(os.path.join(prepath + abspath, img_name))  # cv2读取
                if not isinstance(img, np.ndarray):
                    print("read img error")
                    continue
                img = cv2.resize(img, (224, 224))  # 尺寸变换224*224
                # img = img.astype(np.float32)  # 类型转换为float32
                img = preprocess_input(img)
                label = to_categorical(i, self.num_classes)
                labels.append(label)
                datas.append(img)
        datas = np.array(datas)
        labels = np.array(labels)
        np.save(data_path, datas)
        np.save(label_path, labels)
        return True

    def vgg_model(self):  # 自行构建VGG16
        input_1 = Input(shape=self.shape)  # 输入224*224*3
        # 第一部分
        # 卷积 64深度，大小是3*3 步长为1 使用零填充 激活函数relu
        # 2次卷积 一次池化 池化尺寸2*2 步长2*2
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(input_1)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 64 224*224
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 64 112*112

        # 第二部分 2次卷积 一次池化
        # 卷积 128深度 大小是3*3 步长1 零填充
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 128 112*112
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 128 56*56

        # 第三部分 3次卷积 一次池化 卷积256 3*3
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 256 56*56
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 256 28*28

        # 第四部分 3次卷积 一次池化 卷积 512 3*3
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 512 28*28
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 512 14*14

        # 第五部分 3次卷积 一次池化 卷积 512 3*3
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 512 14*14
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 512 7*7

        x = Flatten()(x)  # 扁平化，用在全连接过渡
        # 第六部分 三个全连接
        # 第一个和第二个全连接相同 输出4096 激活relu 使用dropout，随机丢弃一半
        x = Dense(4096, activation="relu")(x)
        Dropout(0.5)(x)
        x = Dense(4096, activation="relu")(x)
        Dropout(0.5)(x)  # 输出 4096 1*1
        # 第三个全连接层 输出 softmax分类
        out_ = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=input_1, outputs=out_)

        model.summary()

        sgd = optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True)
        model.compile(sgd, loss="categorical_crossentropy", metrics=["accuracy"])


        return model


    def pretrain_vgg(self):  # 采用预训练的VGG16,修改最后一层
        model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))  # 不包含最后一层
        model = Flatten(name='Flatten')(model_vgg.output)
        model = Dense(self.num_classes, activation='softmax')(model)  # 最后一层自定义

        model_vgg = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
        optimizer = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True) # SGD is better than Adam
        # optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model_vgg.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # model_vgg.summary()
        return model_vgg

    def train(self, load_pretrain=False, batch_size=32, epoch=50):
        if load_pretrain:
            model = self.pretrain_vgg()
        else:
            model = self.vgg_model()
        # TensorBoard查看日志
        logs = TensorBoard(log_dir=self.log_path, write_graph=True, write_images=True)

        data_path = self.data_path
        label_path = self.label_path
        save_path = self.model_path
        x = np.load(data_path)
        y = np.load(label_path)
        # 必须打乱 否则验证集loss和acc会出问题
        np.random.seed(200)
        np.random.shuffle(x)
        np.random.seed(200)
        np.random.shuffle(y)
        model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.3,
                  callbacks=[logs])
        model.save(save_path)

    def predict(self, img_path="test.jpg"):
        # model = vgg_model((224,224,3),5)
        model_path = self.model_path
        model = load_model(model_path)
        test_img = cv2.imread(img_path)
        test_img = cv2.resize(test_img, (224, 224))
        test_img = preprocess_input(test_img)
        ans = model.predict(test_img.reshape(1, 224, 224, 3))
        max_index = np.argmax(ans, axis=1)  # 预测结果是值范围0-4的行向量，因此对应的类别序号要+1
        print("预测结果是%s" % (self.classes[max_index[0] + 1]))


data = r"C:\Work\data\train_data.npy"
label = r"C:\Work\data\labels.npy"
mode_path = r"C:\Work\data\flower_photos\flowers_5classes.h5"
vgg16 = vgg((224, 224, 3), 5, data, label, mode_path)
vgg16.generate_data()
vgg16.train(batch_size=32,epoch=50)
# vgg16.predict(imgpath)

#https://www.jianshu.com/p/130171318888

'''