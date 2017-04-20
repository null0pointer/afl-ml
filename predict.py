#!/usr/local/bin/python3

import tensorflow as tf
import math
import file_utils as fu
import ann_utils as ann
import data_utils as du
import matplotlib.pyplot as plt
import sys

TEAM_INDEX = 2
VALUES_START_INDEX = 3

def most_recent_emas(emas, team):
    for i in range(len(emas) - 1, -1, -1):
        ema = emas[i]
        if ema[TEAM_INDEX] == team:
            ema = ema[VALUES_START_INDEX:]
            return ema
    return None

def predict(home, away):
    emas = fu.read_csv(fu.csv_emas_path())
    emas = emas[1:]

    home_ema = most_recent_emas(emas, home)
    away_ema = most_recent_emas(emas, away)

    if home_ema == None or away_ema == None:
        print('Team not found.')
        return

    model_input = [home_ema + away_ema]

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('models/afl.meta')
        saver.restore(sess, 'models/afl')
        Y = sess.run(['Y:0'], {'X:0': model_input, 'Y_:0': [[1,0]], 'pkeep:0': 1.0})
        home_chance = Y[0][0][0]
        away_chance = Y[0][0][1]
        if home_chance > away_chance:
            print('Pick ' + str(home) + ' (' + str(home_chance * 100) + '%)')
        else:
            print('Pick ' + str(away) + ' (' + str(away_chance * 100) + '%)')

if __name__ == '__main__':
    home = sys.argv[-2]
    away = sys.argv[-1]
    predict(home, away)
