import numpy as np
np.random.seed(1111)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential


# data pre-processing
X_train, Y_train = mnist.train.images, mnist.train.labels
X_test, Y_test = mnist.test.images, mnist.test.labels
X_train = X_train.reshape(-1, 1, 28, 28).astype('float32')
X_test = X_test.reshape(-1, 1, 28, 28).astype('float32')

# build model
model = Sequential()

model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',
    input_shape=(1, 28, 28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    border_mode = 'same'
))

model.add(Convolution2D(
    nb_filter=64,
    nb_row=5,
    nb_col=5,
    border_mode='same',))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    border_mode = 'same'
))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))




adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print('training~~~')
model.fit(X_train, Y_train, nb_epoch=3, batch_size=32,)

print('\ntesting~~~')
loss, accuracy = model.evaluate(X_test, Y_test)

print('\ntest loss:', loss)
print('\ntest accuracy:', accuracy)