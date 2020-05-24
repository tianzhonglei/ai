#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
#https://zhuanlan.zhihu.com/p/66182618
#https://www.tensorflow.org/tutorials/keras/regression

from pandas import read_csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#load dataset
dataframe = read_csv("../data/housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
X = dataset[:, 0:13]
Y = dataset[:, 13]
xtrain, xtest, ytrain, ytest=train_test_split(X, Y, test_size=0.15)

#stand data
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

model = Sequential()
model.add(Dense(128, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain, ytrain, batch_size=12, epochs=200, verbose=2)


ypred = model.predict(xtest)
print(model.evaluate(xtrain, ytrain))
#print(ytest)
#print(ypred)

x_ax = range(len(ypred))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


'''  loss, acc with different layers
[0.8525802798049394, 0.05581395348837209]  512 1024 1  

[0.7916680607684823, 0.07906976744186046]  512 512 1

[1.4763440925021505, 0.05116279069767442]  512 128 1

[1.1380963812040727, 0.06744186056907787]  512 256 1

[1.7788248916004978, 0.04186046522024066]  256 256 1

[1.2653102170589359, 0.03255813953488372]  256 512 1

[2.3685778218646383, 0.046511627906976744]  128 256 1

[2.3844684140626775, 0.044186046615589496]  128 128 1

[2.5156423463377844, 0.04186046511627907]   64 128  1

[3.6682540128397387, 0.03255813953488372]   64 64 1

[2.8254008270973383, 0.05116279080163601]   64 64 64 1

[1.031474497706391, 0.06046511627906977]   128 128 128 1

'''