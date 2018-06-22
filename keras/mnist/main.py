import keras
import argparse
from keras.datasets import mnist
from network import ConvNetFunctional, ConvNetSequential


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="sequential", dest="model_type")
    parser.add_argument("--batch_size", type=int, default=128, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=12, dest="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, dest="learning_rate")
    return parser.parse_args()


def get_model(model_type, input_shape, num_classes):
    # get model
    model = None
    if model_type == "sequential":
        model = ConvNetSequential(input_shape, num_classes)
    elif model_type == "functional":
        model = ConvNetFunctional(input_shape, num_classes)
    else:
        print('Please select model_type in ["sequential", "functional"]')
        exit()
    return model


def train_test_MNIST(model_type, batch_size, epochs, lr):
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # get model
    model = get_model(model_type, input_shape, num_classes)

    # set Optimizer and loss function
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # learning
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # test
    score = model.evaluate(x_test, y_test, verbose=0)

    # output
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    args = get_args()
    model_type = args.model_type
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    train_test_MNIST(model_type, batch_size, epochs, lr)


if __name__ == '__main__':
    main()
