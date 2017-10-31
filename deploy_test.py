#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

import papl
import config

sys.dont_write_bytecode = True

argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--test", action="store_true", help="Run test")
argparser.add_argument("-d", "--deploy", action="store_true", help="Run deploy with seven.png")
argparser.add_argument("-s", "--print_syn", action="store_true", help="Print synapses to .syn")
argparser.add_argument("-m", "--model", default="train/model_ckpt_dense", help="Specify a target model file")
args = argparser.parse_args()


def imgread(path):
    img = np.array(Image.open(path).resize((28,28), resample=2))
    return np.reshape(img[:,:,0], (1, 784))


# One option must be set. -t -d -s
if (args.test or args.deploy or args.print_syn) == False:
    argparser.print_help()
    sys.exit()

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

with tf.Session() as sess:
    # Restore values of variables
    saver = tf.train.import_meta_graph(args.model+'.meta')
    saver.restore(sess, args.model)

    # Calc results
    if args.test == True:
        # Evaluate test sets
        accuracy = tf.get_collection("accuracy")[0]

        # To avoid OOM, run validation with 500/10000 test dataset
        start_time = time.time()
        result = 0
        for i in range(20):
            batch = mnist.test.next_batch(500)
            result += sess.run(accuracy, feed_dict={"x:0": batch[0],
                            "y_label:0": batch[1], "keep_prob:0": 1.0})
        result /= 20
        end_time = time.time()

        print("Test accuracy %g" % result)
        print("Time: %s(s)" % (end_time-start_time))
    elif args.deploy == True:
        # Infer a single image & check its latency
        img = imgread('seven.png')
        y_infer = tf.get_collection("y_infer")[0]

        start_time = time.time()
        answer = tf.argmax(y_infer,1)
        result = sess.run(answer, feed_dict={"x:0":img, 
                    "y_label:0":mnist.test.labels, "keep_prob:0": 1.0})
        end_time = time.time()

        print("Output: %s" % result)
        print("Time: %s(s)" % (end_time-start_time))
        papl.log("performance_ref.log", end_time-start_time)

    elif args.print_syn == True:
        # Print synapses (Input data of each neuron)
        img = imgread('seven.png')
        target_syn = config.syn_all
        synapses = [ tf.get_collection(elem.split(".")[0])[0] for elem in target_syn ]
        for i,j in zip(synapses, config.syn_all):
            syn = sess.run(i, feed_dict={"x:0":[img],
                                         "y_:0":mnist.test.labels,
                                         "keep_prob:0": 1.0})
            papl.print_synapse_nps(syn, j)
        print("Done! Synapse data is printed to x.syn")
