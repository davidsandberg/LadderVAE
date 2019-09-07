"""
Variational Autoencoder (VAE) implementation
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
import math

class VAE(object):

    def __init__(self, input_dims, learning_rate, warmup_temp_ph, nrof_stochastic_units, nrof_mlp_units):

        assert len(nrof_stochastic_units)==len(nrof_mlp_units)
        self.nrof_stochastic_units = nrof_stochastic_units
        self.nrof_mlp_units = nrof_mlp_units

        self.input_dims = input_dims

        self.learning_rate = learning_rate
        self.warmup_temp_ph = warmup_temp_ph
        
    def softlimit(self, x, limit=0.1):
        return tf.math.log(tf.math.exp(x) + 1.0 + limit)
    
    def dense(self, x, nrof_units, activation=None, training=True, use_batch_norm=True):
        x = tf.compat.v1.layers.Flatten()(x)
        x = tf.compat.v1.layers.Dense(units=nrof_units)(x)
        if use_batch_norm:
            x = tf.compat.v1.layers.BatchNormalization()(x, training=training)
        x = x if activation is None else activation(x)
        return x
      
    def mlp(self, x, nrof_units, activation, nrof_layers=1, training=True):
        for _ in range(nrof_layers):
            x = self.dense(x, nrof_units=nrof_units, activation=activation, training=training)
        return x

    def sample(self, mu, sigma):
        epsilon = tf.random.normal(tf.shape(mu), mean=0.0, stddev=1.0)
        return mu + sigma*epsilon
        
    def decoder_sample(self, z, mu, sigma, draw_dec_sample=False):
        if draw_dec_sample:
            return self.sample(mu, sigma)
        return z

    def log_normal2(self, x, mean, log_var, eps=1e-5):
        c = - 0.5 * math.log(2*math.pi)
        return c - log_var/2 - (x - mean)**2 / (2 * tf.math.exp(log_var) + eps)
        
    def build_graph(self, x, is_training):
      
        o = dict()
        dbg = dict()
      
        reuse = None if is_training else True
        
        with tf.compat.v1.variable_scope('model', reuse=reuse):
        
            nrof_layers = len(self.nrof_mlp_units)
            
            p_mu, p_sigma = [None]*nrof_layers, [None]*nrof_layers
            q_mu, q_sigma = [None]*nrof_layers, [None]*nrof_layers
            z = [None]*nrof_layers

            # Create encoder
            h = x
            for l in range(nrof_layers):
                dbg['enc_prevh_%d' % l ] = h
                h = self.mlp(h, self.nrof_mlp_units[l], activation=tf.nn.leaky_relu, nrof_layers=nrof_layers, training=is_training)
                dbg['h_%d' % l] = h
      
                """ prepare for bidirectional inference """
                q_mu[l] = self.dense(h, self.nrof_stochastic_units[l], training=is_training)
                q_sigma[l] = self.dense(h, self.nrof_stochastic_units[l], activation=self.softlimit, training=is_training)
                z[l] = self.sample(q_mu[l], q_sigma[l])
                h = z[l]
      
                dbg['z_%d' % l] = z[l]
                dbg['q_mu_%d' % l] = q_mu[l]
                dbg['q_sigma_%d' % l] = q_sigma[l]
      
            # Create decoder
            for l in range(nrof_layers-1, -1, -1):
      
                if l == nrof_layers-1:
                    """ At the highest latent layer, mu & sigma are identical to those output from encoder.
                        And making actual z is not necessary for the highest layer."""
                    mu, sigma = q_mu[l], q_sigma[l]
                    p_mu[l], p_sigma[l] = tf.zeros_like(mu), tf.ones_like(sigma)
                else:
                    """ prior is developed from z of the above layer """
                    p_mu[l] = self.dense(h, self.nrof_stochastic_units[l], training=is_training)
                    p_sigma[l] = self.dense(h, self.nrof_stochastic_units[l], activation=self.softlimit, training=is_training)
      
                dbg['p_mu_%d' % l] = p_mu[l]
                dbg['p_sigma_%d' % l] = p_sigma[l]
                dbg['z_%d' % l]  = z[l]
  
                h = self.mlp(z[l], self.nrof_mlp_units[l], activation=tf.nn.leaky_relu, nrof_layers=nrof_layers, training=is_training)
                dbg['h_%d' % l]  = h
      
            nrof_features = np.prod(self.input_dims)
            x_reconst = self.dense(h, nrof_features, activation=tf.nn.sigmoid, training=is_training, use_batch_norm=False) # Reconstruction

            if len(self.input_dims)==3: 
                x_reconst = tf.reshape(x_reconst, (-1, self.input_dims[0], self.input_dims[1], self.input_dims[2]))
            
            dbg['x_reconst']  = x_reconst
            dbg['x_orig']  = x
    
            # Calculate reconstruction loss
            eps = 1e-5
            x_reconst_clip = tf.clip_by_value(x_reconst, eps, 1-eps)
            log_pxz = -tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_reconst_clip), axis=[1,2])
            o['log_px'] = tf.reduce_mean(log_pxz)
    
            # Calculate ELBO, i.e. log P(X|Z) + temp*(log(P(Z|X)) - log(q(Z)))
            log_qz_list = []
            log_pz_list = []
            for l in range(nrof_layers):
                log_qz_l = self.log_normal2(z[l], q_mu[l], tf.math.log(q_sigma[l])*2)
                log_pz_l = self.log_normal2(z[l], p_mu[l], tf.math.log(p_sigma[l])*2)
                dbg['log_pz_l_%d' % l] = log_pz_l
                dbg['log_qz_l_%d' % l] = log_qz_l
                log_pz_list += [ tf.reduce_sum(log_pz_l, axis=1) ]  # Sum over log probabilities
                log_qz_list += [ tf.reduce_sum(log_qz_l, axis=1) ]  # Sum over log probabilities
            log_pz = tf.add_n(log_pz_list)
            log_qz = tf.add_n(log_qz_list)
            elbo = log_pxz + self.warmup_temp_ph * (log_pz - log_qz)
            
            o['log_pz'] = [ tf.reduce_mean(w) for w in log_pz_list ]  # Average over the batch
            o['log_qz'] = [ tf.reduce_mean(w) for w in log_qz_list ]  # Average over the batch
            
            o['kl_tot'] = -tf.reduce_mean(log_pz - log_qz)  # Average over the batch
            o['kl'] = [ -tf.reduce_mean(lpz-lqz) for (lpz, lqz) in zip(log_pz_list, log_qz_list) ]
                                        
            """ set losses """
            o['elbo'] = tf.reduce_mean(elbo)  # Average over the batch
            
            loss = -o['elbo']
    
            train_op = None
            if is_training:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
                grads = optimizer.compute_gradients(loss)  # Minimize -elbo
                for i,(g,v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_norm(g,5),v) # clip gradients
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(grads)
                
            return train_op, o, dbg
