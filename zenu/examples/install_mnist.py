import tensorflow as tf
import pandas as pd

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images_flattened = train_images.reshape(train_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

train_df = pd.DataFrame(train_images_flattened)
train_df.to_csv('mnist_train_flattened.txt', index=False)

test_df = pd.DataFrame(test_images_flattened)
test_df.to_csv('mnist_test_flattened.txt', index=False)

train_labels_df = pd.DataFrame(train_labels)
train_labels_df.to_csv('mnist_train_labels.txt', index=False)

test_labels_df = pd.DataFrame(test_labels)
test_labels_df.to_csv('mnist_test_labels.txt', index=False)
