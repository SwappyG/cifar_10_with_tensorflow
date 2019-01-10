# For network training
import numpy as np
import tensorflow as tf

from cifar_classes import Dataset

activation_dict = {
    "relu" : tf.nn.relu
}

# Returns appropriate initilizer object based on input data
def learnable_vars_init(data):
    # Get the type of initializer
    type = data["type"]

    # pass in appropriate params based on initilizer
    if (type == "variance_scaling_initializer"):
        return tf.variance_scaling_initializer(scale=data["scale"])

# Defines forward pass of a 2 Conv + 1 FC layer network
def three_layer_conv_net(x, initializer, params, is_training):

    # Get the initializer and activation objects
    init = learnable_vars_init(initializer)
    activation = activation_dict[params["ACTIVATION"]]

    # Ensure that we actually have the objects
    assert(init != None or activation != None)

    # First convolutional layer
    x2 = tf.layers.conv2d(inputs =                x,
                          filters =               params["FILTERS"][0],
                          kernel_size =           params["KERNEL_SIZE"][0],
                          strides =               params["STRIDES"][0],
                          activation =            activation,
                          padding =               "VALID",
                          use_bias =              True,
                          data_format =           "channels_last",
                          kernel_initializer =    init   )

    # Second convolutional layer
    x3 = tf.layers.conv2d(inputs =                x2,
                          filters =               params["FILTERS"][1],
                          kernel_size =           params["KERNEL_SIZE"][1],
                          strides =               params["STRIDES"][1],
                          activation =            activation,
                          padding =               "VALID",
                          use_bias =              True,
                          data_format =           "channels_last",
                          kernel_initializer =    init   )

    # Flatten before fully connected layer
    x3_flat = tf.layers.flatten( x3 )

    # Fully connected layer before output
    scores = tf.layers.dense(inputs =               x3_flat,
                             units =                params["NUM_CLASSES"],
                             use_bias =             True,
                             kernel_initializer =   init )

    return scores
