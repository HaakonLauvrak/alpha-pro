import numpy as np
from tensorflow import keras
import config.config as config


class ANET:
    def __init__(self, name):
        """
        Initialize ANET object.

        Parameters:
        - name (str): The name of the ANET object.
        """
        self.name = name
        if config.game == "nim":
            self.input_shape = (2,)
            self.output_shape = (config.nim_K)
        elif config.game == "hex":
            self.input_shape = (config.board_size, config.board_size, 3)
            self.output_shape = (config.board_size**2)
        self.model = self.build_model()

    def build_model(self):
        """
        Build the ANET model based on the game configuration.

        Returns:
        - model (keras.Model): The built ANET model.
        """
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
                model.add(keras.layers.BatchNormalization())
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
                model.add(keras.layers.BatchNormalization())
            
            model.add(keras.layers.Dense(
                units=self.output_shape, 
                activation="softmax"))
            
        else:
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(shape=self.input_shape))
            for layer in config.dimensions_nim:
                model.add(keras.layers.Dense(
                    units=layer, activation=config.activation))
            model.add(keras.layers.Dense(
                units=self.output_shape, activation="softmax"))
        
        optimizer = keras.optimizers.get({
            'class_name': config.optimizer,
            'config': {'learning_rate': config.learning_rate}})

        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[
                           "accuracy", "mae"])

        return model

    def train_model(self, data) -> float:
        """
        Train the ANET model.

        Parameters:
        - data (dict): A dictionary containing the training data.

        Returns:
        - train_score (float): The training score.
        """
        history = self.model.fit(data["x_train"], data["y_train"],
                       epochs=config.epochs, batch_size=config.batch_size, verbose=1, shuffle=True)
        train_score = self.model.evaluate(
            data["x_train"], data["y_train"], verbose=0)
        
        return train_score

    
    def compute_move_probabilities(self, ann_input):
        """
        Given a state, return a list of probabilities for each move.

        Parameters:
        - ann_input: The input state.

        Returns:
        - probabilities: A list of probabilities for each move.
        """
        return self.model.predict(ann_input, verbose=False)

    def save_model(self, num_episodes):
        """
        Save the ANET model.

        Parameters:
        - num_episodes (int): The number of episodes.
        """
        self.model.save(
            f"actor/weights/{config.game}_{config.board_size}_{num_episodes}ep_{config.num_search_games}searches.keras")
    
    def load_model(self, game, board_size, num_episodes, num_search_games):
        """
        Load a saved ANET model.

        Parameters:
        - game (str): The name of the game.
        - board_size (int): The size of the game board.
        - num_episodes (int): The number of episodes.
        - num_search_games (int): The number of search games.

        Returns:
        - model (keras.Model): The loaded ANET model.
        """
        return keras.models.load_model(f"actor/weights/{game}_{board_size}_{num_episodes}ep_{num_search_games}searches.keras")
   
    def load_model_by_name(self, name):
        """
        Load a saved ANET model by name.

        Parameters:
        - name (str): The name of the model.

        Returns:
        - model (keras.Model): The loaded ANET model.
        """
        return keras.models.load_model(f"actor/weights/{name}.keras")

    def set_model(self, model):
        """
        Set the ANET model.

        Parameters:
        - model (keras.Model): The model to set.
        """
        self.model = model