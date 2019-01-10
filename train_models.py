# General imports
import os

# For network training
import numpy as np
import tensorflow as tf

# For plotting
import matplotlib.pyplot as plt

# For data parsing
from argparse import ArgumentParser
import sys
import json

# functions and classes from other files in project
import init
from data_logging import save_logs
from run_model import run_model


def main():

    # Suppresses some tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("\nThis script will train CNNs to recognize images in the CIFAR-10 dataset\n")

    # Load in a json file and make a dictionary of all our constants
    # This will hold all the parameters for our model
    path = "models/constants_three_layer_conv_net.txt"
    with open(path, 'r') as f:
        CONST = json.load(f)

    print("The following dictionary was loaded into the program:\n")
    print(json.dumps(CONST, indent=4))

    # load the cifar-10 dataset and turn them into batch iterable Dataset objs
    print("\nLoading the CIFAR-10 dataset")
    cifar10_data = init.load_cifar10(
        num_training=CONST["NUM_TRAINING"],
        num_validation=CONST["NUM_VAL"],
        num_test=CONST["NUM_TEST"]
    )
    train_dset, val_dset, test_dset = init.create_dsets(
        cifar10_data,
        CONST["BATCH_SIZE"]
    )

    # Train the model using CONST dictionary on CIFAR-10 datasets
    logs_list = run_model(CONST, train_dset, val_dset, test_dset)

    # Write the logs from the model to file
    save_logs(logs_list)
    print(f"\nDone Execution!\n\n")

if __name__== "__main__":
    main()
