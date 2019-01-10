# For network training
import numpy as np

# Dataset class to hold and iterate through our CIFAR-10 data
# Code borrowed from the Stanford CS231n assignments
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))

class Training_log(object):
    def __init__(self, params):
        self.model = params["model"]
        self.model_params = params["model_params"]
        self.optimizer = params["optimizer"]
        self.optimizer_params = params["optimizer_params"]
        self.epochs = params["epochs"]
        self.iterations = params["iterations"]
        self.batch_size = params["batch_size"]
        self.run_time = params["run_time"]
        self.loss_log = params["loss_log"]
        self.val_log = params["val_log"]
        self.final_val_acc = params["final_val"]
        self.rep_ID = params["rep_ID"]
