"""
Contains some utilities for creating models in tensorflow
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
            with tf.device("/cpu:0"):
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
    Create a 2D convolutional layer with inputs x and filters defined to have
    shape equal to the input shape list.  Weight variables are initialized with
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
    weights = weight_variable(shape, stddev=fan_in)
    biases = bias_variable([shape[-1]])

    return conv1d(x, weights) + biases

def BLSTM(x, size, scope, nonlinearity='logistic'):
    """
    Bidirectional LSTM layer with input vector x and size hidden units.

    Inputs:
        x: Tensor of shape (batch size, time steps, features)
        size: Even integer size of BLSTM layer
        scope: String that sets the variable scope.
        nonlinearity: String specifying activation to use in the RNNs.
                      options include 'logistic' for sigmoid activation and
                      'tanh' for tanh activation.
    Returns:
        Tensor of shape (batch_size, time steps, size) corresponding to the
        output of the BLSTM for each batch at every time step
    """

    # Get the forward input and reverse it for the backwards input
    forward_input = x
    backward_input = tf.reverse(x, [1])

    # Define forward RNN
    with tf.variable_scope('forward_' + scope):
        if nonlinearity == 'logistic':
            forward_lstm = tf.contrib.rnn.BasicLSTMCell(size//2,
                                                        activation=tf.sigmoid)
        elif nonlinearity == 'tanh':
            forward_lstm = tf.contrib.rnn.BasicLSTMCell(size//2,
                                                        activation=tf.tanh)

        forward_out, f_state = tf.nn.dynamic_rnn(forward_lstm, forward_input,
                                                 dtype=tf.float32)

    # Define the backward RNN
    with tf.variable_scope('backward_' + scope):
        if nonlinearity == 'logistic':
            backward_lstm = tf.contrib.rnn.BasicLSTMCell(size//2,
                                                         activation=tf.sigmoid)
        elif nonlinearity == 'tanh':
            backward_lstm = tf.contrib.rnn.BasicLSTMCell(size//2,
                                                         activation=tf.tanh)

        backward_out, b_state = tf.nn.dynamic_rnn(backward_lstm, backward_input,
                                                  dtype=tf.float32)

    # Concatenate the RNN outputs and return
    output = tf.concat([forward_out[:,:,:], backward_out[:,::-1,:]], 2)

    return output
