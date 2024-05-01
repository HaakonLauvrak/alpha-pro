import numpy as np


class REPLAY_BUFFER():

    def __init__(self, max_size):
        self.storage_x_train = []
        self.storage_y_train = []
        self.max_size = max_size

    def add(self, x_train, y_train):
        if len(self.storage_x_train) + len(x_train) >= self.max_size:
            self.storage_x_train[0:(len(x_train) + len(self.storage_x_train)) - self.max_size] = []
            self.storage_y_train[0:(len(y_train) + len(self.storage_y_train)) - self.max_size] = []
        self.storage_x_train.append(x_train)
        self.storage_y_train.append(y_train)

    def get_all(self):
        training_data = {"x_train": np.array(self.storage_x_train), "y_train": np.array(self.storage_y_train)}
        return training_data
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.storage_x_train), size=batch_size)
        x_train = np.array([self.storage_x_train[i] for i in indices])
        y_train = np.array([self.storage_y_train[i] for i in indices])
        training_data = {"x_train": x_train, "y_train": y_train}
        return training_data
    
    def save(self, filename):
        np.savez(filename, x_train=self.storage_x_train, y_train=self.storage_y_train)

    def load(self, filename, append=True):
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
        return len(self.storage_x_train) 