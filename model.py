import tensorflow as tf
import math
from random import randrange
import file_utils as fu
import matplotlib.pyplot as plt

def random_split(input, rate):
    first = []
    second = input
    while len(first) / len(input) < rate:
        index = randrange(0, len(second))
        first.append(second[index])
        second = second[:index] + second[index+1:]
    return first, second

def create_weights_and_biases(layer_sizes):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 2):
        weight = create_weight(layer_sizes[i], layer_sizes[i+1])
        bias = tf.Variable(tf.ones([layer_sizes[i+1]])/layer_sizes[0])
        weights.append(weight)
        biases.append(bias)

    weight = create_weight(layer_sizes[-2], layer_sizes[-1])
    bias = tf.Variable(tf.zeros([layer_sizes[-1]])/layer_sizes[0])
    weights.append(weight)
    biases.append(bias)
    return weights, biases

def link_weights_and_biases(inputs, weights, biases, activation=None, pkeep=None):
    Y = tf.matmul(inputs, weights[0]) + biases[0]
    for i in range(1, len(weights)):
        if not activation == None:
            Y = activation(Y)
        if not pkeep == None:
            Y = tf.nn.dropout(Y, pkeep)
        Y = tf.matmul(Y, weights[i]) + biases[i]
    return Y

def create_weight(in_size, out_size):
    return tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))

def minibatch(data, start, length):
    start_index = start % len(data)
    end_index = start_index + length
    nloops = math.ceil(end_index / len(data))
    looped_data = nloops * data
    return looped_data[start_index:end_index]

raw_inputs = fu.read_csv(fu.csv_inputs_path())
raw_train_ins, raw_test_ins = random_split(raw_inputs, 0.8)

train_inputs = []
train_labels = []
for input in raw_train_ins:
    output = int(float(input[0]))
    label = [0, 0]
    label[output] = 1
    train_labels.append(label)
    ins = [float(x) for x in input[1:]]
    train_inputs.append(ins)

test_inputs = []
test_labels = []
for input in raw_test_ins:
    output = int(float(input[0]))
    label = [0, 0]
    label[output] = 1
    test_labels.append(label)
    ins = [float(x) for x in input[1:]]
    test_inputs.append(ins)

input_size = len(train_inputs[0])
output_size = len(train_labels[0])

X = tf.placeholder(tf.float32, [None, input_size])
Y_ = tf.placeholder(tf.float32, [None, 2])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

layer_sizes = [input_size, 30, 20, 10, output_size]
weights, biases = create_weights_and_biases(layer_sizes)

Ylogits = link_weights_and_biases(X, weights, biases, activation=tf.nn.relu, pkeep=pkeep)
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

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
    train_batch_inputs = minibatch(train_inputs, i * BATCH_SIZE, BATCH_SIZE)
    train_batch_labels = minibatch(train_labels, i * BATCH_SIZE, BATCH_SIZE)

    max_lr = 0.003
    min_lr = 0.0001
    decay_period = 500
    learning_rate = min_lr + (max_lr - min_lr) * math.exp(-i/decay_period)

    if i % 10 == 0 or i == ITERATIONS - 1:
        xaxis.append(i)
        a, c = sess.run([accuracy, cross_entropy], {X: train_inputs, Y_: train_labels, pkeep: 1.0})
        train_accuracies.append(a)
        train_loss.append(c)
        print("Iteration: " + str(i) + "\n\tTrain Accuracy: " + str(a) + " Loss: " + str(c))

        a, c, y = sess.run([accuracy, cross_entropy, Y], {X: test_inputs, Y_: test_labels, pkeep: 1.0})
        test_accuracies.append(a)
        test_loss.append(c)
        print("\t Test Accuracy: " + str(a) + " Loss: " + str(c))

    sess.run(train_step, {X:train_batch_inputs, Y_:train_batch_labels, pkeep:0.65, lr:learning_rate})

plt.plot(xaxis, train_accuracies, 'blue')
plt.plot(xaxis, test_accuracies, 'red')
plt.title('Accuracy')
plt.show()
plt.plot(xaxis, train_loss, 'blue')
plt.plot(xaxis, test_loss, 'red')
plt.title('Loss')
plt.show()
