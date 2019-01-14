# General imports
import os

# For network training
import numpy as np
import tensorflow as tf

# For plotting
import matplotlib.pyplot as plt

# For data parsing
import argparse
import sys
import json

# functions and classes from other files in project
import init
from data_logging import save_logs
from run_model import run_model


def main(**kwargs):

    # Suppresses some tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("\nThis script will train CNNs to recognize images in the CIFAR-10 dataset\n")

    # Load in a json file and make a dictionary of all our constants
    # This will hold all the parameters for our model
    path = "models/" + kwargs.get("load")
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
    if kwargs.get("save") != None:
        folder_name = kwargs.get("save")
        save_logs(logs_list, folder_name)


    print(f"\nDone Execution!\n\n")

if __name__== "__main__":

    description = "This script will train a neural network to recognize images in the CIFAR-10 dataset\n" + \
                  "Define your network as a json file in the models folder\n" + \
                  "Follow 'models/README.txt' and use 'models/model_template.txt' as required\n\n" + \
                  "Define any new networks as functions in 'models.py' using tensorflow OOP API\n" + \
                  "Define any initializer in the 'get_learnable_vars(..) func, and add it to the 'activation dict' in 'models.py'\n" + \
                  "Add your model function to the 'get_scores(..)' function in 'run_model.py'\n\n" + \
                  "Add any loss function in the 'get_loss(..) function in 'run_model.py'\n" + \
                  "Add any optimization function 'get_opt(..) function in 'run_model.py'\n\n" + \
                  "Ensure all the names in the json file match those in the aforementioned functions\n" + \
                  "Call this function with the '-l' or '--load' argument and specify the json file\n"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-l', '--load', required=True)
    parser.add_argument('-s', '--save')

    input_args = parser.parse_args()
    main(**vars(input_args))
