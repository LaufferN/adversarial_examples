import random
import tensorflow as tf
import numpy as np
import sys
import re

from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image

import pickle

def main():

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
    w2 = tf.Variable(tf.random_normal([100, 100], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([100]), name='b2')
    # second hidden layer weights
    w3 = tf.Variable(tf.random_normal([100, 10], stddev=0.03), name='w3')
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
        print(sess.run(accuracy, feed_dict={x_input: test_images, y: test_labels}))

        ############# generate adversary ##############

        # setup new adversarial network

        x_input = tf.Variable(tf.zeros([1,784]))
        # now declare the output data placeholder - 10 digits
        y = tf.placeholder(tf.float32, [None, 10])

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

        img = mnist.test.images[0]

        # trainable_image = tf.Variable(tf.zeros(784))
        x = tf.placeholder(tf.float32, [1,784])
        x_hat = x_input # our trainable adversarial input
        assign_op = tf.assign(x_hat, x)

        learning_rate = tf.placeholder(tf.float32, ())
        y_hat = tf.placeholder(tf.int32, ())

        adversarial_label = tf.one_hot(y_hat, 10)
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
        adversarial_loss  = -tf.reduce_mean(tf.reduce_sum(adversarial_label * tf.log(y_clipped)
                                                  + (1 - adversarial_label) * tf.log(1 - y_clipped), axis=1))
        optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(adversarial_loss, var_list=[x_hat])

        epsilon = tf.placeholder(tf.float32, ())

        below = x - epsilon
        above = x + epsilon
        projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
        with tf.control_dependencies([projected]):
                project_step = tf.assign(x_hat, projected)

        # var_grad = tf.gradients(adversarial_loss, [x_input])
        # grad = sess.run(var_grad, feed_dict={x_input: [img], y_hat: 2})[0][0]
        # epsilon = 10.0/255.0
        # adv = np.add(img, -1 * grad * epsilon)

        demo_epsilon = 255.0/255.0 # perturbation size
        demo_lr = 1e-1
        # demo_lr = .5
        demo_steps = 1000
        demo_target = 2

        # initialization step
        sess.run(assign_op, feed_dict={x: [img]})

        # projected gradient descent
        for i in range(demo_steps):
            # gradient descent step
            _, loss_value = sess.run(
                [optim_step, adversarial_loss],
                feed_dict={learning_rate: demo_lr, y_hat: demo_target})
            # project step
            sess.run(project_step, feed_dict={x: [img], epsilon: demo_epsilon})
            if (i+1) % 10 == 0:
                print('step %d, loss=%g' % (i+1, loss_value))


        adv = x_hat.eval()[0] # retrieve the adversarial example

        np.savetxt('adv_example_vector', adv)

        classif = y_.eval(feed_dict={x_input: [adv]})
        print(np.argmax(classif))
        
        img_orig = Image.new('L', (28,28))
        img_orig.putdata(mnist.test.images[0]*256)
        img_orig.save("orig_image.png")

        img_adv = Image.new('L', (28,28))
        img_adv.putdata(adv*256)
        img_adv.save("adv_image.png")





if __name__ == '__main__':
    main()
