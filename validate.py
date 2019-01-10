import numpy as np
import tensorflow as tf
from cifar_classes import Dataset

# Checks accuracy of network against the specified dataset
# Code borrowed from the Stanford CS231n assignments
def check_accuracy(sess, dset, x, scores, is_training=None):
    """
    Check accuracy on a classification model.

    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """

    num_correct, num_samples = 0, 0

    # Parse through the dataset in batches
    for x_batch, y_batch in dset:

        # Determine the scores without updating weights (no training)
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)

        # Grab the highest score for each sample
        y_pred = scores_np.argmax(axis=1)

        # Add to the running total of samples checked and samples correct
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()

    return num_correct, num_samples
