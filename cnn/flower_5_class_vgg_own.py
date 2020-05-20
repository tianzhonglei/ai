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