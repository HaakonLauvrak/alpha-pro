import numpy as np
from tensorflow import keras
from typing import Dict, List, Any, Union
import config.config as config
import tensorflow as tf


class ANET:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        return keras.Sequential([
            # keras.layers.InputLayer(shape=(config.board_size**2 + 1,)),
            # keras.layers.Dense(input_dim = config.board_size**2 + 1, units=config.board_size**2 + 1, activation="relu"),
            # keras.layers.Dense(units=32, activation="relu"),
            # keras.layers.Dense(units=32, activation="relu"),
            # keras.layers.Dense(units=32, activation="relu"),
            # keras.layers.Dense(units=32, activation="relu"),
            # keras.layers.Dense(units=32, activation="relu"),
            # keras.layers.Dense(units=32, activation="relu"),
            # keras.layers.Dense(units=32, activation="relu"),
            # keras.layers.Dense(units=config.board_size**2, activation="softmax"),
            keras.layers.InputLayer(shape=(2,)),
            keras.layers.Dense(units=8, activation="relu"),
            keras.layers.Dense(units=8, activation="relu"),
            keras.layers.Dense(units=8, activation="relu"),
            keras.layers.Dense(units=5, activation="relu")])
        
    def train_model(self, data) -> float:
        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", "mae"])
        self.model.fit(data["x_train"], data["y_train"], 
                epochs=3, batch_size=20, verbose=False)
        train_score = self.model.evaluate(
                data["x_train"], data["y_train"], verbose=0)
        return train_score[2]


    def compute_move_probabilities(self, state):
        """
        Given a state, return a list of probabilities for each move
        """
        ann_input = state[0].get_ann_input()
        ann_input.append(state[1])
        ann_input = np.array(ann_input)
        ann_input = tf.reshape(ann_input, [1, 2])
        return self.model.predict(ann_input, verbose=False)

    
    