import numpy as np
np.random.seed(1111)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# X shape(60000, 28*28), y shape (10000, )
# (X_train, y_train), (X_test, y_test) = mnist.load_data('MNIST_data')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# data pre-processing
X_train, Y_train = mnist.train.images, mnist.train.labels
X_test, Y_test = mnist.test.images, mnist.test.labels
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Another way to build neural net
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# Another way to define optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, decay=0.0)

model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'], )

model.fit(X_train, Y_train, nb_epoch=10, batch_size=32)

loss, accuracy = model.evaluate(X_test, Y_test)

print('test loss:', loss)
print('test accuracy', accuracy)