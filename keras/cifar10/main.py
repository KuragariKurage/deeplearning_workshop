import keras
import argparse
from network import ConvNetFunctional, ConvNetSequential
from cifar10.data_generator import dataGenerator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="data/train_dataset.txt", dest="train_dataset_path")
    parser.add_argument("--validation_dataset_path", type=str, default="data/validation_dataset.txt", dest="validation_dataset_path")
    parser.add_argument("--test_dataset_path", type=str, default="data/test_dataset.txt", dest="test_dataset_path")
    parser.add_argument("--model_type", type=str, default="sequential", dest="model_type")
    parser.add_argument("--batch_size", type=int, default=128, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=30, dest="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, dest="learning_rate")
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


def train_test_Cifar10(train_dataset_path, validation_dataset_path, test_dataset_path,
                       model_type, batch_size, epochs, lr):

    # set class num
    num_classes = 10

    # set data generator
    train_dataset = dataGenerator(train_dataset_path, num_classes, batch_size, shuffle=True)
    validation_dataset = dataGenerator(validation_dataset_path, num_classes, batch_size, shuffle=True)
    test_dataset = dataGenerator(test_dataset_path, num_classes, batch_size, shuffle=True)

    # get model
    input_shape = train_dataset.get_batch_shape()[1:]
    model = get_model(model_type, input_shape, num_classes)

    # set Optimizer and loss function
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # learning
    model.fit_generator(train_dataset.get_batch(), train_dataset.get_iter_num(), epochs,
                        verbose=1, validation_data=validation_dataset.get_batch(),
                        validation_steps=validation_dataset.get_iter_num(), shuffle=False) # shuffle is implementted in my dataGenerator.

    # test
    score = model.evaluate_generator(test_dataset.get_batch(), steps=test_dataset.get_iter_num(),
                                     verbose=1)

    # output
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    args = get_args()
    train_dataset_path = args.train_dataset_path
    validation_dataset_path = args.validation_dataset_path
    test_dataset_path = args.test_dataset_path
    model_type = args.model_type
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate

    train_test_Cifar10(train_dataset_path, validation_dataset_path, test_dataset_path,
                       model_type, batch_size, epochs, lr)


if __name__ == '__main__':
    main()
