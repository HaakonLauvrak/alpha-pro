import numpy as np


import numpy as np

class REPLAY_BUFFER():
    """
    A class representing a replay buffer for storing and sampling training data.
    """

    def __init__(self, max_size):
        """
        Initializes a new instance of the REPLAY_BUFFER class.

        Parameters:
        - max_size (int): The maximum size of the replay buffer.
        """
        self.storage_x_train = []
        self.storage_y_train = []
        self.max_size = max_size

    def add(self, x_train, y_train):
        """
        Adds training data to the replay buffer.

        Parameters:
        - x_train (list or array-like): The input training data.
        - y_train (list or array-like): The predictions.
        """
        if len(self.storage_x_train) + len(x_train) >= self.max_size:
            self.storage_x_train = self.storage_x_train[1:]
            self.storage_y_train = self.storage_y_train[1:]
            # self.storage_x_train[0:(len(x_train) + len(self.storage_x_train)) - self.max_size] = []
            # self.storage_y_train[0:(len(y_train) + len(self.storage_y_train)) - self.max_size] = []
        self.storage_x_train.append(x_train)
        self.storage_y_train.append(y_train)

    def get_all(self):
        """
        Returns all the training data stored in the replay buffer.

        Returns:
        - training_data (dict): A dictionary containing the input and predictions.
        """
        training_data = {"x_train": np.array(self.storage_x_train), "y_train": np.array(self.storage_y_train)}
        return training_data
    
    def sample(self, batch_size):
        """
        Samples a batch of training data from the replay buffer.

        Parameters:
        - batch_size (int): The size of the batch to sample.

        Returns:
        - training_data (dict): A dictionary containing the input and predictions.
        """
        indices = np.random.randint(0, len(self.storage_x_train), size=batch_size)
        x_train = np.array([self.storage_x_train[i] for i in indices])
        y_train = np.array([self.storage_y_train[i] for i in indices])
        training_data = {"x_train": x_train, "y_train": y_train}
        return training_data
    
    def save(self, filename):
        """
        Saves the training data stored in the replay buffer to a file.

        Parameters:
        - filename (str): The name of the file to save the data to.
        """
        np.savez(filename, x_train=self.storage_x_train, y_train=self.storage_y_train)

    def load(self, filename, append=True):
        """
        Loads training data from a file and adds it to the replay buffer.

        Parameters:
        - filename (str): The name of the file to load the data from.
        - append (bool): Whether to append the loaded data to the existing data in the replay buffer (default: True).
        """
        data = np.load(filename)
        if append:
            if len(self.storage_x_train) == 0:
                self.storage_x_train = data['x_train']
                self.storage_y_train = data['y_train']
            else:
                self.storage_x_train = np.concatenate((self.storage_x_train, data['x_train']), axis=0)
                self.storage_y_train = np.concatenate((self.storage_y_train, data['y_train']), axis=0)
        else:
            self.storage_x_train = data['x_train']
            self.storage_y_train = data['y_train']

    def get_size(self):
        """
        Returns the current size of the replay buffer.
        """
        return len(self.storage_x_train)
