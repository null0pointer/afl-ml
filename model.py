import tensorflow as tf
import math
from random import randrange
import file_utils as fu

def random_split(input, rate):
    first = []
    second = input
    while len(first) / len(input) < rate:
        index = randrange(0, len(second))
        first.append(second[index])
        second = second[:index] + second[index+1:]
    return first, second

raw_inputs = fu.read_csv(fu.csv_inputs_path())
raw_train_ins, raw_test_ins = random_split(raw_inputs, 0.8)

print(len(raw_inputs))

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

print(len(train_inputs))
print(len(test_inputs))

input_size = len(train_inputs[0])
output_size = len(train_labels[0])

X = tf.placeholder(tf.float32, [None, input_size])
Y_ = tf.placeholder(tf.float32, [None, 2])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

layer_sizes = [30, 20, 10]

W1 = tf.Variable(tf.truncated_normal([input_size, layer_sizes[0]], stddev=0.1))
B1 = tf.Variable(tf.ones([layer_sizes[0]])/output_size)
W2 = tf.Variable(tf.truncated_normal([layer_sizes[0], layer_sizes[1]], stddev=0.1))
B2 = tf.Variable(tf.ones([layer_sizes[1]])/output_size)
W3 = tf.Variable(tf.truncated_normal([layer_sizes[1], layer_sizes[2]], stddev=0.1))
B3 = tf.Variable(tf.ones([layer_sizes[2]])/output_size)
W4 = tf.Variable(tf.truncated_normal([layer_sizes[2], output_size], stddev=0.1))
B4 = tf.Variable(tf.zeros([output_size]))

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y1 = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y2 = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y3 = tf.nn.dropout(Y3, pkeep)

Ylogits = tf.matmul(Y3, W4) + B4
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    max_lr = 0.003
    min_lr = 0.0001
    decay_period = 200
    learning_rate = min_lr + (max_lr - min_lr) * math.exp(-i/decay_period)

    a, c = sess.run([accuracy, cross_entropy], {X: train_inputs, Y_: train_labels, pkeep: 1.0})
    print("Epoch: " + str(i) + "\n\tTrain Accuracy: " + str(a) + " Loss: " + str(c))

    a, c = sess.run([accuracy, cross_entropy], {X: test_inputs, Y_: test_labels, pkeep: 1.0})
    print("\t Test Accuracy: " + str(a) + " Loss: " + str(c))

    sess.run(train_step, {X:train_inputs, Y_:train_labels, pkeep:0.75, lr:learning_rate})
