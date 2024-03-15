from tensorflow import keras
from typing import Dict, List, Any, Union
import config.config as config


class ANET:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        return keras.Sequential([
            # keras.layers.InputLayer(input_shape=(config.board_size**2 + 1,)),
            keras.layers.Dense(input_dim = config.board_size**2 + 1, units=config.board_size**2 + 1, activation="relu"),
            # keras.layers.Dense(units=config.board_size**2 + 1, activation="relu"),
            # keras.layers.Dense(units=config.board_size**2 + 1, activation="relu"),
            keras.layers.Dense(units=config.board_size**2, activation="relu"),
        ])
        
    def train_model(self, data) -> float:
        self.model.compile(optimizer=config.optimizer, loss="", metrics=["accuracy"])
        self.model.fit(data["x_train"], data["y_train"], epochs=5)
        return self.model.evaluate(data["x_test"], data["y_test"])[1]

    def compute_move_probabilities(self, state):
        """
        Given a state, return a list of probabilities for each move
        """
        return self.model.predict([state[0].get_cells_as_list(state[1])], verbose=False)

    
    