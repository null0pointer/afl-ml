import tensorflow as tf
import math
import file_utils as fu
import ann_utils as ann
import data_utils as du
import matplotlib.pyplot as plt

raw_inputs = fu.read_csv(fu.csv_inputs_path())
# raw_train_ins, raw_test_ins = du.ordered_split(raw_inputs, 0.8)
raw_train_ins, raw_test_ins = du.random_split(raw_inputs, 0.8)

train_inputs, train_labels = du.split_inputs_and_labels(raw_train_ins, 0, 2)
test_inputs, test_labels = du.split_inputs_and_labels(raw_test_ins, 0, 2)

input_size = len(train_inputs[0])
output_size = len(train_labels[0])

X = tf.placeholder(tf.float32, [None, input_size], name='X')              # Inputs
Y_ = tf.placeholder(tf.float32, [None, 2], name='Y_')                      # Expected outputs
lr = tf.placeholder(tf.float32)                                 # Learning rate

# Dropout keep probability
# DROPOUT_KEEP_RATE = 0.75
DROPOUT_KEEP_RATE = 1.0
pkeep = tf.placeholder(tf.float32, name='pkeep')

layer_sizes = [input_size, 30, 20, 10, output_size]             # ANN architecture
weights, biases = ann.create_weights_and_biases(layer_sizes)    # The weight and bias matrices

Ylogits = ann.link_weights_and_biases(X, weights, biases, activation=tf.nn.relu, pkeep=pkeep)
Y = tf.nn.softmax(Ylogits, name='Y')                                      # Output class probabilities

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)                    # Loss function
cross_entropy = tf.reduce_mean(cross_entropy + 0.01 * ann.l2_regularize_weights(weights), name='cross_entropy')

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

xaxis = []
train_accuracies = []
test_accuracies = []
train_loss = []
test_loss = []

saver = tf.train.Saver(max_to_keep=1)

BATCH_SIZE = 1000
ITERATIONS = 4000
for i in range(ITERATIONS):
    train_batch_inputs = ann.minibatch(train_inputs, i * BATCH_SIZE, BATCH_SIZE)
    train_batch_labels = ann.minibatch(train_labels, i * BATCH_SIZE, BATCH_SIZE)

    max_lr = 0.003
    min_lr = 0.0001
    decay_period = 250
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
        print("\t learning_rate: " + str(learning_rate))

    sess.run(train_step, {X:train_batch_inputs, Y_:train_batch_labels, pkeep:DROPOUT_KEEP_RATE, lr:learning_rate})

saver.save(sess, 'models/afl')

plt.subplot(1,2,1)
plt.plot(xaxis, train_accuracies, 'blue')
plt.plot(xaxis, test_accuracies, 'red')
plt.title('Accuracy')
plt.subplot(1,2,2)
plt.plot(xaxis, train_loss, 'blue')
plt.plot(xaxis, test_loss, 'red')
plt.title('Loss')
plt.show()
