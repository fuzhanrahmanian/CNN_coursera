import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.framework import ops
import cnn_utils as util

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = util.load_dataset()

#an example
index = 0
plt.imshow(X_train_orig[index])
plt.show()
print("y ="+ str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = util.convert_to_one_hot(Y_train_orig, 6).T
Y_test = util.convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])  # None is the batch size
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y

X, Y = create_placeholders(64, 64, 3, 6)


def initialize_parameters():
    tf.set_random_seed(1)
    W1= tf.get_variable("W1", [4, 4, 3, 8], dtype= tf.float32, initializer= tf.layers.xavier_initializer(seed=0))
    W2= tf.get_variable("W1", [2, 2, 8, 16], dtype==tf.float32, initializer=tf.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1, "W2":W2}
    return parameters

tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.reset(init)
    print(parameters["W1"].eval())
    print("W1[1,1,1] = \n" + str(parameters["W1"].eval()[1, 1, 1]))
    print("W1.shape: " + str(parameters["W1"].shape))
    print("\n")
    print("W2[1,1,1] = \n" + str(parameters["W2"].eval()[1, 1, 1]))
    print("W2.shape: " + str(parameters["W2"].shape))

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    Z1 = tf.nn.conv2d(X, W1, strides= [1, 1, 1, 1], padding= 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides= [1, 8, 8, 1], padding= 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides= [1, 1, 1, 1], padding= 'SAME')
    A2= tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides= [1, 4, 4, 1], padding= 'SAME')
    F = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn= None)
    return Z3

tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X, Y= create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X:np.random.randn(2, 64, 64, 3), Y: np.random.randn(2, 6)})
    print("Z3 = \n" + str(a))

def compute_cost(Z3, Y):
    cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= Z3, labels= Y))
    return cost

tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X, Y= create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(Z3, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 5)})
    print("cost = " + str(a))

def model(X_train, Y_train, X_test, Y_test, learning_rate= 0.009,
          num_epochs = 100, minibatch_size= 64, print_cost= True):
    ops.reset_default_graph() # be able to rerun without overwriting
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    X, y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost= 0
            num_minibatches= int(m/minibatch_size)
            seed += 1
            minibatches = util.random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_y})
            if print_cost== True and epoch % 5 == 0:
                print("cost after epoch %i: %f"% (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens')
        plt.title('learning rate= ' + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters

_, _, parameters = model(X_train, Y_train, X_test, Y_test)

# thumbs up .)
fname = "images/thumbs_up.jpg"
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64))
plt.imshow(my_image)