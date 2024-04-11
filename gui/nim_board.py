import numpy as np
import tensorflow as tf
from gui.board import BOARD

class NIM_BOARD(BOARD):

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state
    
    def get_ann_input(self, player):
        ann_input = [self.state]
        ann_input.append(player)
        ann_input = np.array(ann_input)
        ann_input = tf.reshape(ann_input, [1, 2])
        return ann_input