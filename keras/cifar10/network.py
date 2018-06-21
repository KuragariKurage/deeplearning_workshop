from keras.layers import Input, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential, Model

## Convolutional network (sequential version)
# input_shape: input data shape
# n_output: output data shape (number of class)
def ConvNetSequential(input_shape, num_output):
    model = Sequential()

    # CNN model
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_output, activation='softmax'))

    return model

## Convolutional network (functional version)
#  input_shape: input data shape
# n_output: output data shape (number of class)
def ConvNetFunctional(input_shape, num_output):
    # This returns a tensor
    inputs = Input(shape=input_shape)

    # a layer instance is callable on a tensor, and returns a tensor

    x = Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=input_shape, activation='relu')(inputs)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_output, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)

    return model
