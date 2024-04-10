import numpy as np
from tensorflow import keras
from typing import Dict, List, Any, Union
import config.config as config
import tensorflow as tf


class ANET:
    def __init__(self):
        if config.game == "nim":
            self.input_shape = (2,)
            self.output_shape = (config.nim_K)
        elif config.game == "hex":
            self.input_shape = (config.board_size**2 + 1,)
            self.output_shape = (config.board_size**2)
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=self.input_shape))
        for layer in config.dimensions:
            model.add(keras.layers.Dense(units=layer, activation=config.activation))
        model.add(keras.layers.Dense(units=self.output_shape, activation="softmax"))
        return model

    def train_model(self, data) -> float:
        # optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
        optimizer = keras.optimizers.get({
            'class_name': config.optimizer, 
            'config': {'learning_rate': config.learning_rate}})

        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", "mae"])
        self.model.fit(data["x_train"], data["y_train"], 
                epochs=100, batch_size=20, verbose=False)
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
        ann_input = tf.reshape(ann_input, [1, self.input_shape[0]])
        return self.model.predict(ann_input, verbose=False)

    def save_model(self):
        self.model.save(f"actor/weights/{config.game}_{config.num_episodes}episodes_{config.num_search_games}searches.keras")

    def load_model(self, game, num_episodes, num_search_games):
        self.model = keras.models.load_model(f"actor/weights/{game}_{num_episodes}ep_{num_search_games}searches.keras")
