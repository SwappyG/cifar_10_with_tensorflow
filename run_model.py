# For network training
import math
import numpy as np
import tensorflow as tf

import timeit

import models
from cifar_classes import Dataset, Training_log
from validate import check_accuracy

# Returns appropriate optimizer object based on input data
def get_opt(optimizer):
    # Get the optimizer object
    func = optimizer["func"]
    params = optimizer["params"]

    # Instantiate optimizer object with given parameters
    if (optimizer["func"] == "MomentumOptimizer"):
        return tf.train.MomentumOptimizer(
            learning_rate=params["LEARNING_RATE"],
            momentum=params["MOMENTUM"],
            use_nesterov=params["USE_NESTOROV"]
        )

# Grabs the appropriate loss function and returns the loss
def get_loss(loss_func, y, scores, params=None):

    # Call the given loss function with given params
    if (loss_func == "sparse_softmax_cross_entropy_with_logits"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(loss)

    return loss

# Calculate the scores based on a given model and input
def get_scores(model, initializer, x, is_training):

    if model["func"] == "three_layer_conv_net":
        return models.three_layer_conv_net(x, initializer, model["params"], is_training)

def run_model(model, train_dset, val_dset, test_dset):

    start_time = timeit.default_timer()

    logs_list = []

    for rep in range(model["REPS"]):

        print(f"\nRunning rep : {rep+1} of {model['REPS']}\n")
        # Reset the computational graph before we start modelructing one
        print("Resetting the current graph\n")
        tf.reset_default_graph()

        # Determine whether to run on cpu or gpu
        try:
            device = model["DEVICE"]
        except KeyError:
            device = "/cpu:0"

        loss_func_params = None

        # Define the computational graph
        with tf.device( device ):

            # create placeholders to hold minibatches of data
            x = tf.placeholder( tf.float32, [None, model["H"], model["W"], model["C"]] )
            y = tf.placeholder( tf.int32, [None] )

            is_training = tf.placeholder(tf.bool, name='is_training')

            # run the forward pass to get scores
            scores = get_scores(
                model["model"],
                model["initializer"],
                x,
                is_training
            )

            # Calculate the loss
            loss = get_loss(model["loss_func"], y, scores, loss_func_params)

            # Get the appropriate optimizer object
            optimizer = get_opt(model["optimizer"])

            # Add dependancy to force tf.GraphKeys.UPDATE_OPS to run at every step
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Train the model for one step
                train_op = optimizer.minimize(loss)

        # Create a dictionary to log all the data for this run
        log_params = {
            "model" : model["model"]["func"],
            "model_params" : model["model"]["params"],
            "optimizer" : model["optimizer"]["func"],
            "optimizer_params" : model["optimizer"]["params"],
            "epochs" : model["NUM_EPOCHS"],
            "iterations" : model["NUM_ITERS"],
            "batch_size" : model["BATCH_SIZE"],
            "run_time" : 0,
            "loss_log" : [0] * model["NUM_ITERS"] * model["NUM_EPOCHS"],
            "val_log" : [0] * (model["NUM_ITERS"] * model["NUM_EPOCHS"]//model["PRINT_FREQ"] + 1),
            "final_val_acc" : 0,
            "rep_ID" : rep
        }

        this_log = Training_log(log_params)

        if model["NUM_ITERS"] > len(train_dset):
            this_log.num_iters = len(train_dset)

        # Train the model
        print("Training the model")
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # train on the dataset multiple times
            for epoch in range(model["NUM_EPOCHS"]):

                print(f"\nStarting epoch {epoch}\n")

                # Calculate loss and update weights in minibatches
                for t, (x_sub, y_sub) in enumerate(train_dset):

                    # Stop training if t > num_iters specified by user
                    if t >= model["NUM_ITERS"]:
                        break

                    # Collect the relevant info into a dict for loss_step
                    data = {
                        x: x_sub,
                        y : y_sub,
                        is_training:True
                    }

                    # Update the loss using this data batch
                    loss_step, _ = sess.run([loss, train_op], feed_dict=data)

                    # log the loss at this step
                    this_log.loss_log[epoch*model["NUM_ITERS"] + t] = loss_step.item()

                    # Occasionally print status info
                    if t % model["PRINT_FREQ"] == 0:
                        # Check current acc on the val dset
                        num_correct, num_samples = check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                        acc = 100.0*num_correct/num_samples

                        # Print iterations, loss and val_acc
                        print(f"Iteration {t} \t loss = {loss_step} \t val_acc = {acc}")
                        # log the val acc at this step
                        this_log.val_log[ epoch * model["NUM_ITERS"]//model["PRINT_FREQ"]  +  t//model["PRINT_FREQ"]] = acc.item()
                        this_log.final_val_acc = acc.item()

        # Log remainder of data
        this_log.run_time = timeit.default_timer() - start_time
        print(f"\ntotal run time was {this_log.run_time}")

        logs_list.append(this_log)
        del this_log

    return logs_list
