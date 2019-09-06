"""
Train a variational autoencoder on MNIST
"""
# MIT License
# 
# Copyright (c) 2019 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
import argparse
import h5py
from datetime import datetime
import sys, os, time
from VAE import VAE
from LVAE import LVAE
import utils

class Stat(object):

    def __init__(self, filename, ignore_list=[]):
        self.filename = filename
        self.ignore_list = ignore_list
        self.current = dict()
        
    def add(self, new):
        for key, value in new.items():
            if key not in self.ignore_list:
                if key not in self.current:
                    self.current[key] = []
                self.current[key] += [ value ]
        
    def store(self):
        with h5py.File(self.filename, 'w') as f:
            for key, value in self.current.items():
                arr = np.stack(value)
                f.create_dataset(key, data=arr)


def create_dataset(x, y, batch_size):
    # Create a dataset tensor from the images and the labels
    
    # Generator that yields examples from the dataset binarized by sampling
    def gen(x, y):
        for i in range(x.shape[0]):
            xi = x[i,:,:] / 255
            xz = np.random.binomial(1,xi,size=xi.shape)
            yield xz, y[i]
            
    xn = np.expand_dims(x,3)
    dataset = tf.data.Dataset.from_generator(lambda: gen(xn, y), (tf.float32, tf.int32), (tf.TensorShape(xn.shape[1:]), tf.TensorShape([])))
    dataset = dataset.repeat()   # Repeat the dataset indefinitely
    dataset = dataset.shuffle(10000)   # Shuffle the data
    dataset = dataset.batch(batch_size)  # Create batches of data
    dataset = dataset.prefetch(batch_size)  # Prefetch data for faster consumption
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)  # Create an iterator over the dataset

    return iterator
  
def is_nan_or_inf(x_list):
    for x in x_list:
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return True
    return False
  
def get_warmup_temp(epoch, nrof_warmup_epochs):
    if nrof_warmup_epochs>0:
        temp = np.minimum(1.0, 1.0/nrof_warmup_epochs * epoch)
    else:
        temp = 1.0
    return temp
  
def mean(x_list):
    x_mean = dict()
    for q in x_list[0].keys():
        x_mean[q] = np.mean([ a[q] for a in x_list ])
    return x_mean
  
def flatten(x):
    xflat = dict()
    for k, v in x.items():
        if isinstance(v, list):
            for i, zz in enumerate(v):
                xflat['%s_%d' % (k,i)] = zz
        else:
            xflat[k] = v
    return xflat
  
def add_prefix(prefix, x):
    y = dict()
    for k, v in x.items():
        y[prefix+k] = v
    return y
  
def to_list(strng):
    return [ int(x) for x in strng.split(',') ]
        
  
def main(args):

    src_path,_ = os.path.split(os.path.realpath(__file__))

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    res_dir = os.path.join(os.path.expanduser(args.output_base_dir), subdir)
    if not os.path.isdir(res_dir):  # Create the log directory if it doesn't exist
        os.makedirs(res_dir)
        
    # Store some git revision info in a text file in the log directory
    utils.store_revision_info(src_path, res_dir, ' '.join(sys.argv))
    
    # Store parameters in an HDF5 file
    utils.store_hdf(os.path.join(res_dir, 'parameters.h5'), vars(args))
    
    # Create statistics object
    stat = Stat(os.path.join(res_dir, 'stat.h5'))

    with tf.Graph().as_default():
        tf.compat.v1.random.set_random_seed(args.seed)
        np.random.seed(args.seed)
    
        ###########################################
        """             Load Data               """
        ###########################################
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        nrof_train_batches = int(np.ceil(x_train.shape[0] / args.batch_size))
        nrof_test_batches = int(np.ceil(x_test.shape[0] / args.batch_size))
        input_dims = (x_train.shape[1], x_train.shape[2], 1)
        train_iterator = create_dataset(x_train, y_train, args.batch_size)
        test_iterator = create_dataset(x_test, y_test, args.batch_size)
        xtrain, ytrain = train_iterator.get_next() #@UnusedVariable
        xtest, ytest = test_iterator.get_next() #@UnusedVariable
    
        ###########################################
        """        Build Model Graphs           """
        ###########################################
        with tf.compat.v1.variable_scope("vae"):
    
            warmup_temp = tf.compat.v1.placeholder(tf.float32, shape=(), name="warmup_temp")
    
            if args.model_type=='VAE':
                m = VAE(input_dims, args.learning_rate, warmup_temp, to_list(args.nrof_stochastic_units), to_list(args.nrof_mlp_units))
            elif args.model_type=='LVAE':
                m = LVAE(input_dims, args.learning_rate, warmup_temp, to_list(args.nrof_stochastic_units), to_list(args.nrof_mlp_units))
            else:
                raise ValueError('Invalid model type')
            print('Building train graph...')
            train_op, train_o, train_dbg = m.build_graph(xtrain, is_training=True)

            print('Building evaluation graph...')
            _, eval_o, eval_dbg = m.build_graph(xtest, is_training=False) #@UnusedVariable

        init_op = tf.compat.v1.global_variables_initializer()
    
        sess  = tf.compat.v1.InteractiveSession()
        sess.run(init_op)
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
    
        print('... start training')
        for epoch in range(1, args.nrof_epochs+1):
    
            # Get warm-up temperature
            temp = get_warmup_temp(epoch, args.nrof_warmup_epochs)
    
            start_time = time.time()
            for _ in range(nrof_train_batches):
                feed_dict = {warmup_temp: temp}
                o, dbg, _ = sess.run([train_o, train_dbg, train_op], feed_dict=feed_dict) #@UnusedVariable
                stat.add(add_prefix('train_', flatten(o)))
                
                #if is_nan_or_inf(dbg.values()) or is_nan_or_inf(o.values()):
                #    xxx = 1 #@UnusedVariable
    
            print(' epoch: %5d  time: %6.3f   temp: %10.3f  elbo: %10.3f   log p(x): %10.3f   log p(z): %8.3f | %8.3f  log q(z): %8.3f | %8.3f  KL(q(z|x)||p(z)): %8.3f | %8.3f' % \
                  (epoch, time.time()-start_time, temp, o['elbo'], o['log_px'], o['log_pz'][0], o['log_pz'][1], o['log_qz'][0], o['log_qz'][1], o['kl'][0], o['kl'][1] ))
            
            # Evaluate every n epochs
            if epoch % args.eval_every_n_epochs == 0:
                o_list = []
                start_time = time.time()
                for _ in range(nrof_test_batches):
                    feed_dict = {warmup_temp: 1.0}
                    o, dbg = sess.run([eval_o, eval_dbg], feed_dict=feed_dict) #@UnusedVariable
                    o_list += [ flatten(o) ]
        
                o_mean = mean(o_list)
                
                stat.add(add_prefix('eval_', o_mean))
                if args.display_eval:
                    print('*epoch: %5d  time: %6.3f   temp: %10.3f  elbo: %10.3f   log p(x): %10.3f   log p(z): %8.3f | %8.3f  log q(z): %8.3f | %8.3f  KL(q(z|x)||p(z)): %8.3f | %8.3f' % \
                          (epoch, time.time()-start_time, 1.0, o_mean['elbo'], o_mean['log_px'], o_mean['log_pz_0'], o_mean['log_pz_1'], o_mean['log_qz_0'], o_mean['log_qz_1'], o_mean['kl_0'], o_mean['kl_1'] ))

            # Store statistics
            stat.store()
                

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_base_dir', type=str, 
        help='Directory where output folders are stored.', default='./out/')
    parser.add_argument('--nrof_epochs', type=int,
        help='Number of epochs to run.', default=300)
    parser.add_argument('--seed', type=int,
        help='Random seed for numpy and tensorflow.', default=666)
    parser.add_argument('--batch_size', type=int,
        help='Number of examples to process in a batch.', default=256)
    parser.add_argument('--learning_rate', type=float,
        help='Learning rate before decaying.', default=0.002)
    parser.add_argument('--nrof_warmup_epochs', type=int,
        help='Number of epochs to do warm-up.', default=100)
    parser.add_argument('--model_type', type=str,
        help='Model type.', default='VAE')
    parser.add_argument('--nrof_stochastic_units', type=str,
        help='Number of units in each stochastic layer.', default='64,32')
    parser.add_argument('--nrof_mlp_units', type=str,
        help='Number of units in each multi-layer perceptron.', default='512,256')
    parser.add_argument('--eval_every_n_epochs', type=int,
        help='Evaluate every n epochs.', default=1)
    parser.add_argument('--ladder_share_params', 
        help='Parameters are shared between the encoder and decoder in the LVAE.', action='store_true')
    parser.add_argument('--display_eval', 
        help='Display results from evaluation.', action='store_true')
    
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    
