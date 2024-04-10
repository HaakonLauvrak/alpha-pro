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
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=config.board_size**2, activation="softmax"),
        ])
        
    def train_model(self, data) -> float:
        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", "mae"])
        history = self.model.fit(data["x_train"], data["y_train"], epochs=10, batch_size=20, verbose=False, validation_split=0.1)
        train_score = self.model.evaluate(data["x_train"], data["y_train"], verbose=0)
        val_score = history.history['val_accuracy'][-1]
        return train_score, val_score


    def compute_move_probabilities(self, state):
        """
        Given a state, return a list of probabilities for each move
        """
        return self.model.predict([state[0].get_cells_as_list(state[1])], verbose=False)

    def save_model(self):
        self.model.save(f"{config.game}_{config.num_episodes}episodes_{config.num_search_games}searchgames.h5")

    def load_model(self, game, num_episodes, num_search_games):
        self.model = keras.models.load_model(f"{game}_{num_episodes}episodes_{num_search_games}searchgames.h5")