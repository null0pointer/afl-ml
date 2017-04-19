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

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('models/afl.meta')
    saver.restore(sess, 'models/afl')

    a, c = sess.run(['accuracy:0', 'cross_entropy:0'], {'X:0': test_inputs, 'Y_:0': test_labels, 'pkeep:0': 1.0})
    print("Test Accuracy: " + str(a) + " Loss: " + str(c))
