# For network training
import numpy as np
import tensorflow as tf

from cifar_classes import Dataset

# Loads the cifar10 dataset
# Code borrowed from the Stanford CS231n assignments
def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test

# turns cifar10 tensors in Dataset objects for easier batch iterations
def create_dsets(cifar10_data, batch_size):
    X_train, y_train, X_val, y_val, X_test, y_test = cifar10_data

    # Let the user know the shapes of the data as a sanity check
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape, y_train.dtype)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Create Dataset objects for easy iteration through our data
    train_dset = Dataset(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )
    val_dset = Dataset(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False
    )
    test_dset = Dataset(
        X_test,
        y_test,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dset, val_dset, test_dset
