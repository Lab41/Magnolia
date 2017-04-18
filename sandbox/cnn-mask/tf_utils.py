"""
Contains some utilities for creating convolutional networks in tensorflow
"""

import functools
import tensorflow as tf

def scope_decorator(function):
    """
    Decorator that handles graph construction and variable scoping
    """

    name = function.__name__
    attribute = '_cache_' + name

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self,attribute):
            with tf.device("/gpu:0"):
                with tf.variable_scope(name):
                    setattr(self,attribute,function(self))
        return getattr(self,attribute)

    return decorator

def weight_variable(shape, stddev):
    """
    Creates a variable tensor with a shape defined by the input

    Inputs:
        shape: list containing dimensionality of the desired output
        stddev: standard deviation to initialize with
    """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, value=0.1):
    """
    Creates a variable tensor with dimensionality defined by the input and
    initializes it to a constant

    Inputs:
        shape: list containing dimensionality of the desired output
        value: float specifying the initial value of the variable
    """
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)

def leaky_relu(x, alpha=0.1):
    """
    Leaky rectified linear unit.  Returns max(x, alpha*x)
    """
    return tf.maximum(x, alpha*x)

def conv2d(x, W):
    """
    Performs a 2D convolution over inputs x using filters defined by W.  All
    strides are set to one and padding is 'SAME'.
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def conv2d_layer(x, shape):
    """
    Create a 2D convolutional layer with inputs x and filters defined to have shape
    equal to the input shape list.  Weight variables are initialized with
    standard deviations set to account for the fan in.
    """
    fan_in = tf.sqrt(2/(shape[2] + shape[3]))
    weights = weight_variable(shape, stddev=fan_in)
    biases = bias_variable([shape[-1]])

    return conv2d(x, weights) + biases

def conv1d(x, W):
    """
    Performs a 1D convolution over inputs x using filters defined by W.  All
    strides are set to one and padding is 'SAME'.
    """
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def conv1d_layer(x, shape):
    """
    Create a 1D convolutional layer with inputs x and filters defined to have
    shape equal to the input shape list.  Weight variables are initialized with 
    standard deviations set to account for the fan in.
    """
    fan_in = tf.sqrt(2/(shape[1] + shape[2]))
    weights = weight_variable(shape)
    biases = bias_variable([shape[-1]])

    return conv1d(x, weights) + biases
