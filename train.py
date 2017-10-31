#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import config
import papl

sys.dont_write_bytecode = True

argparser = argparse.ArgumentParser()
argparser.add_argument("-1", "--first_round", action="store_true",
    help="Run 1st-round: train with 20000 iterations")
argparser.add_argument("-2", "--second_round", action="store_true",
    help="Run 2nd-round: apply pruning and its additional training")
argparser.add_argument("-3", "--third_round", action="store_true",
    help="Run 3rd-round: transform model to a sparse format and save it")
argparser.add_argument("-m", "--checkpoint", default="train/model_ckpt_dense",
    help="Target checkpoint model file for 2nd and 3rd round")
args = argparser.parse_args()

# One option must be set
if (args.first_round or args.second_round or args.third_round) == False:
    argparser.print_help()
    sys.exit()

def apply_prune(tensors):
    """Pruning with given weight.
    
    Args:
        tensors: Tensor dict.
    Returns:
        pruning index dict.
    """
    # Store nonzero index for each weights.
    dict_nzidxs = {}

    # For each target layers,
    for target in config.target_layer:
        wl = "w_" + target
        print(wl + " threshold:\t" + str(config.th[wl]))
        # Get target layer's weights
        tensor = tensors[wl]
        weight = tensor.eval()

        # Apply pruning
        weight, nzidxs = papl.prune_dense(weight, name=wl,
                                            thresh=config.th[wl])

        # Store pruned weights as tensorflow objects
        dict_nzidxs[wl] = nzidxs
        sess.run(tensor.assign(weight))

    return dict_nzidxs


def apply_prune_on_grads(grads_and_vars, dict_nzidxs):
    """Apply pruning on gradients.
    Mask gradients with pruned elements.

    Args:
        grads_and_vars: computed gradient by Optimizer.
        dict_nzidxs: dictionary for each tensor with nonzero indexs.
    
    Returns:
    
    """
    # For each pruned weights with nonzero index list,
    for key, nzidxs in dict_nzidxs.items():
        count = 0
        # For each gradients and variables,
        for grad, var in grads_and_vars:
            # Find matched tensor
            if var.name == key+":0":
                nzidx_obj = tf.cast(tf.constant(nzidxs), tf.float32)
                grads_and_vars[count] = (tf.multiply(nzidx_obj, grad), var)
            count += 1

    return grads_and_vars


def generate_sparse_dict(dense_tensors):
    """Generate sparse dictionary.

    Args:
        dense_tensors: weight with dense form.

    Returns:
        sparse_tensors: tensor with sparse form.
    """
    sparse_tensors = {}
    for target in config.target_all_layer:
        weight = np.transpose(dense_tensors[target].eval())
        indices, values, shape = papl.prune_sparse(weight, name=target)
        sparse_tensors[target+"_idx"] = tf.Variable(indices, 
                dtype=tf.int32, name=target+"_idx")
        sparse_tensors[target] = tf.Variable(values, dtype=tf.float32, name=target)
        sparse_tensors[target+"_shape"] = tf.Variable(shape, 
                dtype=tf.int32, name=target+"_shape")
    return sparse_tensors


def dense_cnn_model(x, weights):
    """Make dense CNN model.

    Args:
        x(tensor): input tensor with shape [-1, 28, 28, 1].
        weights: 

    Returns:
        logits layer.
    """
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, weights["w_conv1"]) + weights["b_conv1"])
    tf.add_to_collection("in_conv1", x_image)
    h_pool1 = max_pool_2x2(h_conv1)
    tf.add_to_collection("in_conv2", h_pool1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["w_conv2"]) + weights["b_conv2"])
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    tf.add_to_collection("in_fc1", h_pool2_flat)

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights["w_fc1"]) + weights["b_fc1"])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    tf.add_to_collection("in_fc2", h_fc1_drop)

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, weights["w_fc2"]) + weights["b_fc2"])
    return y_conv


def test(y_infer, y_label, message="None."):
    """Test.

    Args:
        y_infer: inference result.
        y_label: label.

    Returns:
        proportion for correct prediction.
    """
    correct_prediction = tf.equal(tf.argmax(y_infer, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # To avoid OOM, run validation with 500/10000 test dataset
    result = 0
    for i in range(20):
        batch = mnist.test.next_batch(500)
        result += accuracy.eval(feed_dict={x: batch[0], y_label: batch[1], keep_prob: 1.0})
    result /= 20

    print(message+" %g\n" % result)
    return result


def check_file_exists(key):
    fileList = os.listdir(".")
    count = 0
    for elem in fileList:
        if elem.find(key) >= 0:
            count += 1
    return key + ("-"+str(count) if count>0 else "")


# Read dataset
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Construct a dense model
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_label = tf.placeholder(tf.float32, shape=[None, 10], name="y_label")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

dense_w={
    "w_conv1": tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1), name="w_conv1"),
    "b_conv1": tf.Variable(tf.constant(0.1,shape=[32]), name="b_conv1"),
    "w_conv2": tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1), name="w_conv2"),
    "b_conv2": tf.Variable(tf.constant(0.1,shape=[64]), name="b_conv2"),
    "w_fc1": tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1), name="w_fc1"),
    "b_fc1": tf.Variable(tf.constant(0.1,shape=[1024]), name="b_fc1"),
    "w_fc2": tf.Variable(tf.truncated_normal([1024,10],stddev=0.1), name="w_fc2"),
    "b_fc2": tf.Variable(tf.constant(0.1,shape=[10]), name="b_fc2")
}

y_infer = dense_cnn_model(x, dense_w)
tf.add_to_collection("y_conv", y_infer)

train_dir = config.train_dir
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
saver = tf.train.Saver()

if args.first_round == True:
    # First round: Train baseline dense model
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_infer)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_infer,1), tf.argmax(y_label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.add_to_collection("accuracy", accuracy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_label: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_label: batch[1], keep_prob: 0.5})

        # Test
        score = test(y_infer, y_label, message="First-round prune-only test accuracy")
        papl.log("baseline_accuracy.log", score)
        
        # Save model objects to readable format
        papl.print_weight_vars(dense_w, papl.config.target_all_layer,
                               papl.config.target_dat, show_zero=papl.config.show_zero)
        # Save model objects to serialized format
        saver.save(sess, os.path.join(train_dir, "model_ckpt_dense"))

if args.second_round == True:
    # Second round: Retrain pruned model, start with default model: model_ckpt_dense
    with tf.Session() as sess:
        saver.restore(sess, args.checkpoint)

        # Apply pruning on this context
        dict_nzidx = apply_prune(dense_w)

        # save model objects to readable format
        papl.print_weight_vars(dense_w, papl.config.target_all_layer,
                               papl.config.target_p_dat, show_zero=papl.config.show_zero)

        # Test prune-only networks
        score = test(y_infer, y_label, message="Second-round prune-only test accuracy")
        papl.log("prune_accuracy.log", score)

        # save model objects to serialized format
        saver.save(sess, os.path.join(train_dir, "model_ckpt_dense_pruned"))

        # Retrain networks
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_infer)
        trainer = tf.train.AdamOptimizer(1e-4)
        # Compute gradient and remove change for pruning weight
        grads_and_vars = trainer.compute_gradients(cross_entropy)
        grads_and_vars = apply_prune_on_grads(grads_and_vars, dict_nzidx)
        train_step = trainer.apply_gradients(grads_and_vars)

        correct_prediction = tf.equal(tf.argmax(y_infer, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize firstly touched variables (mostly from accuracy calc.)
        for var in tf.global_variables():
            if tf.is_variable_initialized(var).eval() == False:
                sess.run(tf.variables_initializer([var]))

        # Train x epochs additionally
        for i in range(config.retrain_iterations):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_label: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_label: batch[1], keep_prob: 0.5})

        # Save retrained variables to a desne form
        saver.save(sess, os.path.join(train_dir, "model_ckpt_dense_retrained"))

        # Test the retrained model
        score = test(y_infer, y_label, message="Second-round final test accuracy")
        papl.log("final_accuracy.log", score)

if args.third_round == True:
    # Third round: Transform iteratively pruned model to a sparse format and save it
    with tf.Session() as sess:
        if args.second_round == False:
            saver.restore(sess, os.path.join(train_dir, "model_ckpt_dense_pruned"))

        # Transform final weights to a sparse form
        sparse_w = generate_sparse_dict(dense_w)

        # Initialize new variables in a sparse form
        for var in tf.global_variables():
            if tf.is_variable_initialized(var).eval() == False:
                sess.run(tf.variables_initializer([var]))

        # Save model objects to readable format
        papl.print_weight_vars(dense_w, config.target_all_layer,
                               config.target_tp_dat, show_zero=config.show_zero)

        # Save model objects to serialized format
        final_saver = tf.train.Saver(sparse_w)
        final_saver.save(sess, os.path.join(train_dir, "model_ckpt_sparse_retrained"))
