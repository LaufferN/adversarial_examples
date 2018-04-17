import random
import tensorflow as tf
import numpy as np
import sys
import re

from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image

import pickle

def adv():

    # hyperparameters
    learning_rate = 0.5
    epochs = 10
    batch_size = 100

    # grab mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_labels = mnist.train.labels
    train_images = mnist.train.images
    train_pca_images = []

    # same procedure for test labels
    test_labels = mnist.test.labels
    test_images = mnist.test.images


    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x_input = tf.placeholder(tf.float32, [None, 784])

    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    # declare the weights connecting the input to the hidden layer
    w1 = tf.Variable(tf.random_normal([784, 100], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random_normal([100]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    w2 = tf.Variable(tf.random_normal([100, 5], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([5]), name='b2')
    # second hidden layer weights
    w3 = tf.Variable(tf.random_normal([5, 10], stddev=0.03), name='w3')
    b3 = tf.Variable(tf.random_normal([10]), name='b3')

    # calculate the output of the hidden layer
    hidden_1 = tf.add(tf.matmul(x_input, w1), b1)
    hidden_1 = tf.nn.relu(hidden_1)

    hidden_out = tf.add(tf.matmul(hidden_1, w2), b2)
    hidden_out = tf.nn.relu(hidden_out)

    # hidden_out = tf.add(tf.matmul(hidden_2, w3), b3)
    # hidden_out = tf.nn.relu(hidden_out)

    # calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w3), b3))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf_prediction = 0 # used to check correct behavior


    # start the session
    with tf.Session() as sess:
        # initialize the variables
        sess.run(init_op)
        # grab a batch
        total_batch = int(len(mnist.train.labels) / batch_size)
        print("Starting training")
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x = train_images[i*batch_size : (i+1)*batch_size]
                batch_y = train_labels[i*batch_size : (i+1)*batch_size]
                # feed batches
                _, c = sess.run([optimizer, cross_entropy], feed_dict={x_input: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

        print("\nTraining complete!")
        accuracy_ratio = sess.run(accuracy, feed_dict={x_input: test_images, y: test_labels})
        print(accuracy_ratio)

        ############# generate adversary ##############

        # recover adv example

        adv = np.loadtxt('adv_example_vector')

        classif = y_.eval(feed_dict={x_input: [adv]})
        print(np.argmax(classif))

        return (np.argmax(classif) == 7, accuracy_ratio)
        
def main(steps):
    sum_ = 0
    accuracy_sum = 0
    i = 0
    while i < 50 :
        adv_result = adv()
        if adv_result[1] < .9:
            continue # without increment 
        if adv_result[0]:
            sum_ += 1
        accuracy_sum += adv_result[1]
        i += 1

    print("Ratio of misclasifications: " + str(1 - float(sum_)/float(steps)))
    print("Average accuracy: " + str(float(accuracy_sum)/float(steps)))

    

if __name__ == '__main__':

    main(50)
