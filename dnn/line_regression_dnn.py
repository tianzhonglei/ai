import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)# 数据随机化
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))# 创建数据及参数, 并加入噪声

# 绘制数据
#plt.scatter(X, Y)
#plt.show()

# 分为训练数据和测试数据
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

model = Sequential()
model.add(Dense(units=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train,Y_train, epochs=500, verbose=0)
print(model.predict([10.0]))
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test loss: ', cost)

# 获取参数
#W, b = model.layers[0].get_weights()
#print('Weights: ',W)
#print('Biases: ', b)