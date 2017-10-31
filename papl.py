#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import csv
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages

import config

sys.dont_write_bytecode = True


def _save_to_pdf(output):
    """Save output to pdf file."""
    pp = PdfPages(output)
    #plt.savefig(pp, format='pdf')
    pp.close()
    #plt.close()


def _to_percent(y, position):
    """Manipulate y-axis of histogram."""
    percent_y = str(y*100)
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return percent_y + r'$\%$'
    else:
        return percent_y + '%'


def _min_ruler(array):
    """Calculate min position of each histogram bin."""
    minimum = min(array)
    print(" - min: ", minimum)
    offset = minimum % step         # ???
    return minimum - offset


def _max_ruler(array):
    """Calculate max position of each histogram bin."""
    maximum = max(array)
    print(" - max: ", maximum)
    offset = maximum % step         # ???
    return maximum - offset + step


def draw_histogram(*target, **kwargs):
    """Draw histogram.
    Histogram settings are configurable through config.py.

    Args:
        target(option): data from global variables (config) or arguments.
    Returns:
        histogram.(x.pdf)
    """
    if len(target) == 1:
        target = target[0]
        assert type(target) == list
        file_list = target
    else:
        file_list = config.weight_all

    # Define global variable.
    global step
    step = kwargs["step"]

    for filename in file_list:
        print("Target: ", filename)
        try:
            with open(config.pdf_prefix+"%s" % filename) as text:
                x = np.float32(text.read().rstrip("\n").split("\n"))

            norm = np.ones_like(x)
            # Make axis
            binspace = np.arange(_min_ruler(x), _max_ruler(x), step)
            # Draw histogram
            #n, bins, patches = plt.hist(x, bins=binspace, weights=norm,
            #    alpha=config.alpha, facecolor=config.color)

            #plt.grid(True)
            _save_to_pdf(config.pdf_prefix+"%s.pdf" % filename.split(".")[0])
        except IOError as e:
            print("Warning: I/O error(%d) - %s" % (e.errno, e.strerror))
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
    print("Graphs are drawned!")


def print_weight_vars(obj_dict, weight_obj_list, fname_list, show_zero=False):
    """Print weight variables.
    
    Args:
        model object list
    Returns:
        human-readable form of model as 'x.dat'
    """
    train_dir = config.train_dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    for elem, fname in zip(weight_obj_list, fname_list):
        weight_arr = obj_dict[elem].eval()
        ndim = weight_arr.size
        flat_weight_space = weight_arr.reshape(ndim)
        with open(os.path.join(train_dir, fname), "w") as filelog:
            if show_zero == False:
                flat_weight_space = flat_weight_space[flat_weight_space != 0]
            writeLine = csv.writer(filelog, delimiter='\n')
            writeLine.writerow(flat_weight_space)


def print_sparse_weight_vars(obj_dict, weight_obj_list, fname_list):
    """Print sparse matrix.

    Args:
        sparse model object list
    Returns:
        human-readable form of model as 'x.dat'
    """
    train_dir = config.train_dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    for elem, fname in zip(weight_obj_list, fname_list):
        weight_arr = obj_dict[elem].eval().values
        ndim = weight_arr.size
        flat_weight_space = weight_arr.reshape(ndim)
        with open(os.path.join(train_dir, fname), "w") as filelog:
            writeLine = csv.writer(filelog, delimiter='\n')
            writeLine.writerow(flat_weight_space)


def print_synapse_nps(syn_arr, fname, show_zero=False):
    """Print synapse.

    Args:
        synapse
    Returns:
        human-readable form of model as x.syn
    """
    train_dir = config.train_dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    ndim = syn_arr.size
    flat_syn_space = syn_arr.reshape(ndim)
    with open(os.path.join(train_dir, fname), "w") as filelog:
        if show_zero == False:
            flat_syn_space = flat_syn_space[flat_syn_space != 0]
        writeLine = csv.writer(filelog, delimiter='\n')
        writeLine.writerow(flat_syn_space)


def prune_dense(weight_arr, name="None", thresh=0.005):
    """Apply weight pruning with threshold with dense matrix form.
    
    Args:
        weight_arr: n-d dense array.

    Returns:
        indices: index list of non-zero elements
        values: value list of non-zero elements
        shape: original shape of matrix
    """
    assert isinstance(weight_arr, np.ndarray)
    
    under_threshold = abs(weight_arr) < thresh
    over_threshold = abs(weight_arr) >= thresh
    weight_arr[under_threshold] = 0

    # Count zero elements.
    count = np.sum(under_threshold)
    print("Non-zero count (Dense %s): %d" % (name, weight_arr.size - count))
    return weight_arr, over_threshold


def prune_sparse(weight_arr, name="None", thresh=0.005):
    """Apply weight pruning with threshold with sparse matrix form.

    Args:
        weight_arr: anonymous dimension array.
        thresh: pruning threshold.

    Returns:
        indices: index list of non-zero elements
        values: value list of non-zero elements
        shape: original shape of matrix
    """
    assert isinstance(weight_arr, np.ndarray)

    under_threshold = abs(weight_arr) < thresh
    weight_arr[under_threshold] = 0

    # Make matrix as sparse form.
    values = weight_arr[weight_arr != 0]
    indices = np.transpose(np.nonzero(weight_arr))
    shape = list(weight_arr.shape)

    # Count zero elements.
    count = np.sum(under_threshold)
    print("Non-zero count (Sparse %s): %d" % (name, weight_arr.size - count))
    return indices, values, shape


def log(fname, log):
    train_dir = config.train_dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    with open(os.path.join(train_dir, fname), "a") as f:
        f.write(str(log)+"\n")


def imread(path):
    """Read image and resize it to (28, 28).

    Args:
        path: file path.

    Returns:
        ndarray resized to fixed (28, 28).
    """
    return np.array(Image.open(path).resize((28,28), resample=2))
