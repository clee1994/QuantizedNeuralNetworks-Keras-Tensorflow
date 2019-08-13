# -*- coding: utf-8 -*-
import numpy as np

from keras import backend as K

from keras.layers import InputSpec, Layer, Dense, Conv2D
from keras import constraints
from keras import initializers

#from quantized_ops import quantize, clip_through

from __future__ import absolute_import
import keras.backend as K
import tensorflow as tf
import numpy as np




def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)
    rounded_through = x + K.stop_gradient(rounded - x)
    return rounded_through


def clip_through(x, min_val, max_val):
    '''Element-wise clipping with gradient propagation
    Analogue to round_through
    '''
    clipped = K.clip(x, min_val, max_val)
    clipped_through= x + K.stop_gradient(clipped-x)
    return clipped_through 


def clip_through(x, min, max):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,min,max)
    return x + K.stop_gradient(clipped - x)


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return K.clip((x+1)/2, 0, 1)




def quantize(W, nb = 16, clip_through=False):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    if clip_through:
        Wq = clip_through(round_through(W*m),-m,m-1)/m
    else:
        Wq = K.clip(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq


def quantized_relu(W, nb=16):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    #non_sign_bits = nb-1
    #m = pow(2,non_sign_bits)
    #Wq = K.clip(round_through(W*m),0,m-1)/m

    nb_bits = nb
    Wq = K.clip(2. * (round_through(_hard_sigmoid(W) * pow(2, nb_bits)) / pow(2, nb_bits)) - 1., 0,
                1 - 1.0 / pow(2, nb_bits - 1))
    return Wq


def quantized_tanh(W, nb=16):

    '''The weights' binarization function,

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = K.clip(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq

def quantized_leakyrelu(W, nb=16, alpha=0.1):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    if alpha != 0.:
        negative_part = tf.nn.relu(-W)
    W = tf.nn.relu(W)
    if alpha != 0.:
        alpha = tf.cast(tf.convert_to_tensor(alpha), W.dtype.base_dtype)
        W -= alpha * negative_part

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = clip_through(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq

def quantized_maxrelu(W, nb=16):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    non_sign_bits = nb-1
    max_ = tf.reduce_max((W))
    #max_ = tf.Print(max_,[max_])
    max__ = tf.pow(2.0,tf.ceil(tf.log(max_)/tf.log(tf.cast(tf.convert_to_tensor(2.0), W.dtype.base_dtype))))
    #max__ = tf.Print(max__,[max__])
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = max__*clip_through(round_through(W/max__*(m)),0,m-1)/(m)
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq

def quantized_leakymaxrelu(W, nb=16, alpha=0.1):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    if alpha != 0.:
        negative_part = tf.nn.relu(-W)
    W = tf.nn.relu(W)
    if alpha != 0.:
        alpha = tf.cast(tf.convert_to_tensor(alpha), W.dtype.base_dtype)
        W -= alpha * negative_part

    max_ = tf.reduce_max((W))
    #max_ = tf.Print(max_,[max_])
    max__ = tf.pow(2.0,tf.ceil(tf.log(max_)/tf.log(tf.cast(tf.convert_to_tensor(2.0), W.dtype.base_dtype))))
    #max__ = tf.Print(max__,[max__])

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = max__* clip_through(round_through(W/max__*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)

    return Wq



def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = quantize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb


class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        #todo: switch for clip through?
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"name": self.__call__.__name__,
                "min_value": self.min_value,
                "max_value": self.max_value}


class QuantizedDense(Dense):
    ''' Binarized Dense layer
    References: 
    "QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, units, H=1., nb=16, kernel_lr_multiplier='Glorot', bias_lr_multiplier=None, **kwargs):
        super(QuantizedDense, self).__init__(units, **kwargs)
        self.H = H
        self.nb = nb
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        super(QuantizedDense, self).__init__(units, **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot H: {}'.format(self.H))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot learning rate multiplier: {}'.format(self.kernel_lr_multiplier))
            
        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
        quantized_kernel = quantize(self.kernel, nb=self.nb)
        output = K.dot(inputs, quantized_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)


        return output
        
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantizedConv2D(Conv2D):
    '''Binarized Convolution2D layer
    References: 
    "QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, filters, kernel_regularizer=None,activity_regularizer=None, kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier=None, H=1., nb=16,  **kwargs):
        super(QuantizedConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.nb = nb
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        self.activity_regularizer =activity_regularizer
        self.kernel_regularizer = kernel_regularizer
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
            
        base = self.kernel_size[0] * self.kernel_size[1]
        if self.H == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.H = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
            #print('Glorot H: {}'.format(self.H))
            
        if self.kernel_lr_multiplier == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5/ (nb_input + nb_output)))
            #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        #self.bias_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight((self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        quantized_kernel = quantize(self.kernel, nb=self.nb)

        inverse_kernel_lr_multiplier = 1./self.kernel_lr_multiplier
        inputs_qnn_gradient = (inputs - (1. - 1./inverse_kernel_lr_multiplier) * K.stop_gradient(inputs))\
                  * inverse_kernel_lr_multiplier

        outputs_qnn_gradient = K.conv2d(
            inputs_qnn_gradient,
            quantized_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        outputs = (outputs_qnn_gradient - (1. - 1./self.kernel_lr_multiplier) * K.stop_gradient(outputs_qnn_gradient))\
                  * self.kernel_lr_multiplier


        #outputs = outputs*K.mean(K.abs(self.kernel))

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases

QuantizedConvolution2D = QuantizedConv2D
