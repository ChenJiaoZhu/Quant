# Coding: UTF-8

from keras.models import Sequential
from keras import losses
from keras.layers import Dense, Activation, Dropout

def ann(x_train, x_test, y_train, y_test)
    model = Sequential()
    model.add(Dense(256, input_dim=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.summary()
    model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=1,
                        verbose=1, validation_data=(x_test,y_test))
    return history


