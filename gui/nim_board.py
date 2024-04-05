import numpy as np
from gui.board import BOARD

class NIM_BOARD(BOARD):

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state
    
    def get_ann_input(self):
        return [self.state]