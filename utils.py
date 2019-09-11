"""
Utility functions
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

import os
import datetime
from subprocess import Popen, PIPE
import pickle
import shutil
import h5py
import numpy as np

def gettime():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')


def store_revision_info(src_path, output_dir, arg_string):
  
    # Get git hash
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_hash = stdout.strip()
  
    # Get local changes
    gitproc = Popen(['git', 'diff', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_diff = stdout.strip()
      
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('command line: %s\n--------------------\n' % arg_string)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def get_learning_rate_from_file(filename, step):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                s = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if s <= step:
                    learning_rate = lr
                else:
                    return learning_rate
                  
def copy_learning_rate_schedule_file(filename, target_dir):
    shutil.copy(filename, target_dir)
    new_path = os.path.join(target_dir, os.path.split(filename)[1])
    return new_path
                  
def load_pickle(filename):
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
    return arr
  
def save_pickle(filename, var_list):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)
        
def store_hdf(filename, stat):
    with h5py.File(filename, 'w') as f:
        for key, value in stat.items():
            f.create_dataset(key, data=value)
            
class Stat(object):

    def __init__(self, filename, ignore_list=[]):
        self.ignore_list = ignore_list
        self.current = dict()
        self.filename = filename
        self.is_created = False
        
    def add(self, new):
        for key, value in new.items():
            if key not in self.ignore_list:
                if key not in self.current:
                    self.current[key] = []
                self.current[key] += [ value ]

    def store(self):
        flag = 'a' if self.is_created else 'w'
        with h5py.File(self.filename, flag) as f:
            for key, value in self.current.items():
                if value:  # Do not try to add empty lists
                    arr = np.hstack(value)
                    if not key in f.keys():
                        maxshape = (None,) + arr.shape[1:]
                        f.create_dataset(key, arr.shape, chunks=arr.shape, maxshape=maxshape)
                    else:
                        new_size = f[key].shape[0] + arr.shape[0]
                        f[key].resize(new_size, axis=0)
                    f[key][-arr.shape[0]:] = arr
                    self.current[key] = []
            self.is_created = True

    def load(self):
        with h5py.File(self.filename, 'r') as f:
          for key in f.keys():
              self.current[key] = np.array(f[key])
              #print('%s: %s' % (key, self.current[key].shape))
        return self
