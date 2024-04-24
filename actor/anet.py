import numpy as np
from tensorflow import keras
from typing import Dict, List, Any, Union
import config.config as config
import tensorflow as tf


class ANET:
    def __init__(self, name):
        self.name = name
        if config.game == "nim":
            self.input_shape = (2,)
            self.output_shape = (config.nim_K)
        elif config.game == "hex":
            self.input_shape = (config.board_size, config.board_size, 3)
            self.output_shape = (config.board_size**2)
        self.model = self.build_model()

    def build_model(self):
        # Convolutional neural network
        if config.game == "hex":
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(
                shape=self.input_shape))
            
            for layer in config.dimensions_conv:
                model.add(keras.layers.Conv2D(
                    filters=layer,
                    kernel_size=(3, 3),
                    activation=config.activation,
                    padding="same",
                    strides=(1,1),
                    ))
            model.add(keras.layers.Conv2D(
                filters = 1, 
                kernel_size = (1,1),
                activation = config.activation, 
                strides = (1,1)
            ))
            model.add(keras.layers.Flatten())
            for layer in config.dimensions_dense:
                model.add(keras.layers.Dense(
                    units=layer, 
                    activation=config.activation
                ))
            
            model.add(keras.layers.Dense(
                units=self.output_shape, 
                activation="softmax",
                ))
            return model
            
        #Traditional neural network
        else:
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(shape=self.input_shape))
            for layer in config.dimensions_nim:
                model.add(keras.layers.Dense(
                    units=layer, activation=config.activation))
            model.add(keras.layers.Dense(
                units=self.output_shape, activation="softmax"))
            return model

    def train_model(self, data) -> float:
        # optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
        optimizer = keras.optimizers.get({
            'class_name': config.optimizer,
            'config': {'learning_rate': config.learning_rate}})

        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[
                           "accuracy", "mae"])
        history = self.model.fit(data["x_train"], data["y_train"],
                       epochs=config.epochs, batch_size=config.batch_size, verbose=True, shuffle=True)
        train_score = self.model.evaluate(
            data["x_train"], data["y_train"], verbose=0)
        return train_score

    def compute_move_probabilities(self, ann_input):
        """
        Given a state, return a list of probabilities for each move
        """
        return self.model.predict(ann_input, verbose=False)

    def save_model(self, num_episodes):
        self.model.save(
            f"actor/weights/{config.game}_{config.board_size}_{num_episodes}ep_{config.num_search_games}searches.keras")
    
    def load_model(self, game, board_size, num_episodes, num_search_games):
        return keras.models.load_model(f"actor/weights/{game}_{board_size}_{num_episodes}ep_{num_search_games}searches.keras")

    def set_model(self, model):
        self.model = model