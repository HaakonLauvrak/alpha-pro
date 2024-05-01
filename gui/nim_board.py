import numpy as np
import tensorflow as tf
from gui.board import BOARD

class NIM_BOARD(BOARD):
    """
    A class representing a NIM board.
    """

    def set_state(self, state):
        """
        Sets the state of the NIM board.
        """
        self.state = state

    def get_state(self):
        """
        Returns the current state of the NIM board.
        """
        return self.state
    
    def get_ann_input(self, player):
        """
        Returns the input for the artificial neural network (ANN).

        Args:
            player (int): The current player.
        """
        ann_input = [self.state]
        ann_input.append(player)
        ann_input = np.array(ann_input)
        ann_input = tf.reshape(ann_input, [1, 2])
        return ann_input