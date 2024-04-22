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
        self.storage_x_train += x_train
        self.storage_y_train += y_train

    def get_all(self):
        training_data = {"x_train": self.storage_x_train, "y_train": self.storage_y_train}
        return training_data
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.storage_x_train), size=batch_size)
        x_train = np.array([self.storage_x_train[i] for i in indices])
        y_train = np.array([self.storage_y_train[i] for i in indices])
        training_data = {"x_train": x_train, "y_train": y_train}
        return training_data
    
    def save(self, filename):
        np.savez(filename, x_train=self.storage_x_train, y_train=self.storage_y_train)

    def load(self, filename):
        data = np.load(filename)
        self.storage_x_train = data['x_train']
        self.storage_y_train = data['y_train']

    def get_size(self):
        return len(self.storage_x_train) 