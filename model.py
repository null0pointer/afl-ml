import tensorflow as tf
import math
from random import randrange
import file_utils as fu
import ann_utils as ann
import data_utils as du
import matplotlib.pyplot as plt

raw_inputs = fu.read_csv(fu.csv_inputs_path())
raw_train_ins, raw_test_ins = du.ordered_split(raw_inputs, 0.8)

train_inputs, train_labels = du.split_inputs_and_labels(raw_train_ins, 0, 2)
test_inputs, test_labels = du.split_inputs_and_labels(raw_test_ins, 0, 2)

input_size = len(train_inputs[0])
output_size = len(train_labels[0])

X = tf.placeholder(tf.float32, [None, input_size])              # Inputs
Y_ = tf.placeholder(tf.float32, [None, 2])                      # Expected outputs
lr = tf.placeholder(tf.float32)                                 # Learning rate
pkeep = tf.placeholder(tf.float32)                              # Dropout keep probability

layer_sizes = [input_size, 30, 20, 10, output_size]             # ANN architecture
weights, biases = ann.create_weights_and_biases(layer_sizes)    # The weight and bias matrices

Ylogits = ann.link_weights_and_biases(X, weights, biases)#, activation=tf.nn.relu, pkeep=pkeep)
Y = tf.nn.softmax(Ylogits)                                      # Output class probabilities

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100             # Loss function

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

xaxis = []
train_accuracies = []
test_accuracies = []
train_loss = []
test_loss = []

BATCH_SIZE = 1000
ITERATIONS = 4000
for i in range(ITERATIONS):
    train_batch_inputs = ann.minibatch(train_inputs, i * BATCH_SIZE, BATCH_SIZE)
    train_batch_labels = ann.minibatch(train_labels, i * BATCH_SIZE, BATCH_SIZE)

    max_lr = 0.003
    min_lr = 0.0001
    decay_period = 400
    learning_rate = min_lr + (max_lr - min_lr) * math.exp(-i/decay_period)

    if i % 10 == 0 or i == ITERATIONS - 1:
        xaxis.append(i)
        a, c = sess.run([accuracy, cross_entropy], {X: train_inputs, Y_: train_labels, pkeep: 1.0})
        train_accuracies.append(a)
        train_loss.append(c)
        print("Iteration: " + str(i) + "\n\tTrain Accuracy: " + str(a) + " Loss: " + str(c))

        a, c = sess.run([accuracy, cross_entropy], {X: test_inputs, Y_: test_labels, pkeep: 1.0})
        test_accuracies.append(a)
        test_loss.append(c)
        print("\t Test Accuracy: " + str(a) + " Loss: " + str(c))

    sess.run(train_step, {X:train_batch_inputs, Y_:train_batch_labels, pkeep:0.65, lr:learning_rate})

plt.subplot(1,2,1)
plt.plot(xaxis, train_accuracies, 'blue')
plt.plot(xaxis, test_accuracies, 'red')
plt.title('Accuracy')
plt.subplot(1,2,2)
plt.plot(xaxis, train_loss, 'blue')
plt.plot(xaxis, test_loss, 'red')
plt.title('Loss')
plt.show()
