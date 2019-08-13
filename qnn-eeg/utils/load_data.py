# Copyright 2017 Bert Moons

# This file is part of QNN.

# QNN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# QNN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# The code for QNN is based on BinaryNet: https://github.com/MatthieuCourbariaux/BinaryNet

# You should have received a copy of the GNU General Public License
# along with QNN.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
# from pylearn2.datasets.cifar10 import CIFAR10
# from pylearn2.datasets.mnist import MNIST

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator



def load_dataset(dataset):
    if (dataset == 'CIFAR-10'):
        print('Loading CIFAR 10')
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
        train_dat_size = 45000
        train_dat = x_train[0:train_dat_size,:,:]
        train_label = y_train[0:train_dat_size]
        valid_dat = x_train[train_dat_size:-1,:,:]
        valid_label = y_train[train_dat_size:-1]
        test_dat = x_test
        test_label = y_test



        train_dat = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., train_dat), 1.), (-1, 3, 32, 32)),(0,2,3,1))
        valid_dat = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., valid_dat), 1.), (-1, 3, 32, 32)),(0,2,3,1))
        test_dat = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., test_dat), 1.), (-1, 3, 32, 32)),(0,2,3,1))
        
        # convert class vectors to categorical
        num_classes = y_test.max()-y_test.min()+1
        train_label = keras.utils.to_categorical(train_label,num_classes)
        valid_label = keras.utils.to_categorical(valid_label,num_classes)
        test_label = keras.utils.to_categorical(test_label,num_classes)

        # convert to +-1 labels (for hinge loss for some reason)

        num_classes= num_classes*2-1   
        train_label= train_label*2-1
        valid_label= valid_label*2-1
        test_label = test_label *2-1

        # for the time being we do not do any data augmentation - might do it later if required
        train_set = (train_dat,train_label)
        valid_set = (valid_dat,valid_label)
        test_set  = (test_dat,test_label)




#    if (dataset == "CIFAR-10"):
#
#        print('Loading CIFAR-10 dataset...')
#
#        train_set_size = 45000
#        train_set = CIFAR10(which_set="train", start=0, stop=train_set_size)
#        valid_set = CIFAR10(which_set="train", start=train_set_size, stop=50000)
#        test_set = CIFAR10(which_set="test")
#
#        train_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., train_set.X), 1.), (-1, 3, 32, 32)),(0,2,3,1))
#        valid_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., valid_set.X), 1.), (-1, 3, 32, 32)),(0,2,3,1))
#        test_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., test_set.X), 1.), (-1, 3, 32, 32)),(0,2,3,1))
#        # flatten targets
#        train_set.y = np.hstack(train_set.y)
#        valid_set.y = np.hstack(valid_set.y)
#        test_set.y = np.hstack(test_set.y)
#
#        # Onehot the targets
#        train_set.y = np.float32(np.eye(10)[train_set.y])
#        valid_set.y = np.float32(np.eye(10)[valid_set.y])
#        test_set.y = np.float32(np.eye(10)[test_set.y])
#
#        # for hinge loss
#        train_set.y = 2 * train_set.y - 1.
#        valid_set.y = 2 * valid_set.y - 1.
#        test_set.y = 2 * test_set.y - 1.
#        # enlarge train data set by mirrroring
#        x_train_flip = train_set.X[:, :, ::-1, :]
#        y_train_flip = train_set.y
#        train_set.X = np.concatenate((train_set.X, x_train_flip), axis=0)
#        train_set.y = np.concatenate((train_set.y, y_train_flip), axis=0)
#
#    elif (dataset == "MNIST"):
#
#        print('Loading MNIST dataset...')
#
#        train_set_size = 50000
#        train_set = MNIST(which_set="train", start=0, stop=train_set_size)
#        valid_set = MNIST(which_set="train", start=train_set_size, stop=60000)
#        test_set = MNIST(which_set="test")
#
#        train_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., train_set.X), 1.), (-1, 1, 28, 28)),(0,2,3,1))
#        valid_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., valid_set.X), 1.), (-1, 1,  28, 28)),(0,2,3,1))
#        test_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., test_set.X), 1.), (-1, 1,  28, 28)),(0,2,3,1))
#        # flatten targets
#        train_set.y = np.hstack(train_set.y)
#        valid_set.y = np.hstack(valid_set.y)
#        test_set.y = np.hstack(test_set.y)
#
#        # Onehot the targets
#        train_set.y = np.float32(np.eye(10)[train_set.y])
#        valid_set.y = np.float32(np.eye(10)[valid_set.y])
#        test_set.y = np.float32(np.eye(10)[test_set.y])
#
#        # for hinge loss
#        train_set.y = 2 * train_set.y - 1.
#        valid_set.y = 2 * valid_set.y - 1.
#        test_set.y = 2 * test_set.y - 1.
#        # enlarge train data set by mirrroring
#        x_train_flip = train_set.X[:, :, ::-1, :]
#        y_train_flip = train_set.y
#        train_set.X = np.concatenate((train_set.X, x_train_flip), axis=0)
#        train_set.y = np.concatenate((train_set.y, y_train_flip), axis=0)
#



    else:
        print("wrong dataset given")

    return train_set, valid_set, test_set
