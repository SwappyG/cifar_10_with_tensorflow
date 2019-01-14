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

    def __len__(self):
        return len(self.X[0])

class Training_log(object):
    def __init__(self, params):
        self.model = params.get("model")
        self.model_params = params.get("model_params")
        self.optimizer = params.get("optimizer")
        self.optimizer_params = params.get("optimizer_params")
        self.epochs = params.get("epochs")
        self.iterations = params.get("iterations")
        self.batch_size = params.get("batch_size")
        self.run_time = params.get("run_time")
        self.loss_log = params.get("loss_log")
        self.val_log = params.get("val_log")
        self.final_val_acc = params.get("final_val_acc")
        self.rep_ID = params.get("rep_ID")
