import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)
loss, acc = model.evaluate(x_test, y_test)
print(f"Loss: {loss} , Accuracy: {acc}")

prediction = model.predict([x_test])

for i in range(2):
    plt.figure(figsize=(15,15))
    correct = 0
    for l in range(25):
        plt.subplot(5, 5, l+1)
        k = (i+1) * (25+l)
        plt.imshow(x_test[k], cmap=plt.cm.binary)
        plt.xlabel("Prediction: {} - Real Data: {}".format(np.argmax(prediction[k]), y_test[k]))
        plt.xticks([])
        plt.yticks([])
        if np.argmax(prediction[k]) == y_test[k]:
            plt.ylabel("Correct")
            correct += 1
        else:
            plt.ylabel("Wrong")

    print("Accuracy for the set of 25 images: {}".format(correct/25))
    plt.show()
    time.sleep(5)
    sys.exit()
