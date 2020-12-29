from copy import copy

import tensorflow
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt

import Utils

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
clothes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class CNN:
    def __init__(self):
        self.filters = [32]  # , 64]
        self.kernel_size = [(2, 2)]  # , (3, 3)]
        self.activations = ['relu']  # , 'tanh']
        self.neurons = [128]  # , 256]
        self.models = ['getModel1']  # , 'getModel2']
        self.getModel = {'getModel1': self.getModel1}  # ,
        # 'getModel2': self.getModel2
        # }
        self.best_model = self.BestModel()

    class BestModel:
        def __init__(self):
            self.accuracy = 0
            self.filter = 0
            self.kernel_size = 0
            self.neurons = 0
            self.activation = ''
            self.functionName = ''
            self.model = None
            self.cnnsDescription = {'getModel1': f'one convolution and one max pooling layers',
                                    'getModel2': f'two convolution layers, second layer have twice as many filters,\n'
                                                 f'two layer of max pooling and one dropout layer'}

        def updateValues(self, accuracy, newFilter, kernel_size, activation, neurons, functionName, model):
            if self.accuracy < accuracy:
                self.accuracy = accuracy
                self.filter = newFilter
                self.kernel_size = kernel_size
                self.activation = activation
                self.neurons = neurons
                self.functionName = functionName
                self.model = copy(model)

        def __str__(self):
            return f"This is CNN with {self.cnnsDescription[self.functionName]}\n" \
                   f"Accuracy: {self.accuracy}\n" \
                   f"Filters: {self.filter}\n" \
                   f"Kernel size: {self.kernel_size}\n" \
                   f"Activation function: {self.activation}\n" \
                   f"Neurons on last layer: {self.neurons}\n"

    def getModel1(self, filters, kernel_size,
                  activation1, activation2,
                  input_shape, pixelW,
                  pixelH, neurons, classes):
        model = keras.Sequential([
            keras.layers.Conv2D(filters,
                                kernel_size=kernel_size,
                                activation=activation1,
                                input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(input_shape=(pixelW, pixelH)),
            keras.layers.Dense(neurons, activation=activation1),
            keras.layers.Dense(classes, activation=activation2)
        ])
        return model

    def getModel2(self, filters, kernel_size,
                  activation1, activation2,
                  input_shape, pixelW,
                  pixelH, neurons, classes):
        model = keras.Sequential([
            keras.layers.Conv2D(filters,
                                kernel_size=kernel_size,
                                activation=activation1,
                                input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(2 * filters,
                                kernel_size=kernel_size,
                                activation=activation1,
                                input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Dropout(0.5),
            keras.layers.Flatten(input_shape=(pixelW, pixelH)),
            keras.layers.Dense(neurons, activation=activation1),
            keras.layers.Dense(classes, activation=activation2)
        ])
        return model

    def getBestModel(self, trainX, trainY, testX, testY, labels):
        shape = trainX.shape
        for m in self.models:
            print(m)
            for f in self.filters:
                for kernel_size in self.kernel_size:
                    for activation in self.activations:
                        for n in self.neurons:
                            model = self.getModel[m](f, kernel_size, activation,
                                                     'softmax', (shape[1], shape[2], 1),
                                                     shape[1], shape[2], n, labels)
                            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                            model.fit(trainX, trainY, batch_size=32, validation_data=(testX, testY), verbose=0)
                            testLoss, testAcc = model.evaluate(testX, testY, verbose=2)
                            self.best_model.updateValues(testAcc, f, kernel_size, activation, n, m, model)

        Utils.file_print('Best model for MNIST.txt', self.best_model)
        # падает тут
        predicted = self.best_model.model.predict(testX)
        denormalize_predicted = []
        for i in predicted:
            denormalize_predicted.append(numpy.argmax(i))
        denormalize_predicted = numpy.array(denormalize_predicted, float)
        confusion_matrix = tensorflow.math.confusion_matrix(testY, denormalize_predicted)

        Utils.file_print('MNIST confusion matrix.txt', confusion_matrix)
        Utils.print_cool_matrix(numbers, (trainX, trainY), predicted, "Numbers")

        return self.best_model


def normalize(arr1, arr2, arr3, arr4):
    return (numpy.array(arr1 / 255.0, float), arr2), (numpy.array(arr3 / 255.0, float), arr4)


def doMNIST(cnn):
    (digitsX, digitsY), (testDigitsX, testDigitsY) = keras.datasets.mnist.load_data()
    labels = max(digitsY) + 1

    (digitsX, digitsY), (testDigitsX, testDigitsY) = (digitsX.reshape(digitsX.shape[0], 28, 28, 1),
                                                      keras.utils.to_categorical(digitsY)), \
                                                     (testDigitsX.reshape(testDigitsX.shape[0], 28, 28, 1),
                                                      keras.utils.to_categorical(testDigitsY))

    (digitsX, digitsY), (testDigitsX, testDigitsY) = normalize(digitsX, digitsY, testDigitsX, testDigitsY)
    bestModel = cnn.getBestModel(digitsX, digitsY, testDigitsX, testDigitsY, labels)
    return bestModel


def doFashion(bestModel, cnn):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    model = cnn.getModel[bestModel.functionName](
        bestModel.filter, bestModel.kernel_size,
        bestModel.activation, 'softmax',
        'input_shape', 'pixelW', 'pixelH',
        bestModel.neurons, 'classes')

    model.fit()
    testLoss, testAcc = model.evaluate()


def main():
    # cnn = CNN()
    # best_model = doMNIST(cnn)
    # doFashion(best_model, cnn)

    (digitsX, digitsY), (testDigitsX, testDigitsY) = keras.datasets.mnist.load_data()
    cool_matrix = []
    for i in range(10):
        cool_matrix.append([digitsX[i]] * 10)

    plt.figure(figsize=(80, 80))
    fig, ax = plt.subplots(nrows=10, ncols=10)
    # position = numpy.array(10)
    # ax.set_xticks(position)
    # ax.set_xticklabels(numbers)
    # ax.set_yticks(position)
    # ax.set_yticklabels(numbers)
    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(cool_matrix[i][j])
    plt.show()


main()
