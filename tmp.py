import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

new_train_imgs = []
for i in train_images:
    new_train_imgs.append(i / 255.0)

new_test_imgs = []
for i in test_images:
    new_test_imgs.append(i / 255.0)

new_train_images = numpy.array(new_train_imgs, float)
new_test_images = numpy.array(new_test_imgs, float)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
# model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(new_train_images, train_labels, epochs=10, batch_size=32, validation_data=(new_test_images, test_labels), verbose=0)

test_loss, test_acc = model.evaluate(new_test_images, test_labels, verbose=0)
print("acc=", test_acc, sep="")

predict = model.predict(test_images)
arr = []
for i in predict:
    arr.append(numpy.argmax(i))

predict_labels = numpy.array(arr, float)

cm = tensorflow.math.confusion_matrix(test_labels, predict_labels)

print("\nCONFUSION MATRIX")
print(cm)