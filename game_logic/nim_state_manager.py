import copy
from game_logic.state_manager import STATE_MANAGER
import config.config as config

class NIM_STATE_MANAGER(STATE_MANAGER):

    
    def getLegalMoves(self, state):
        return [i for i in range(1, (min(config.nim_K, state[0].get_state())) + 1)]
    
    def makeMove(self, move, state):
        state[0].set_state(state[0].get_state() - move)
        state[1] = 1 if state[1] == -1 else -1
    
    def simulateMove(self, move, state):
        new_state = copy.deepcopy(state)
        new_state[0].set_state(new_state[0].get_state() - move)
        new_state[1] = 1 if new_state[1] == -1 else -1
        return new_state
    
    def isGameOver(self, state):
        return state[0].get_state() <= 0
    
    def getReward(self, state):
        return -state[1]
    
    def find_all_moves(self):
        return [i for i in range(1, config.nim_K + 1)]
    
    def getLegalMovesList(self, state):
        legal_moves = self.getLegalMoves(state)
        return [1 if i in legal_moves else 0 for i in range(1, config.nim_K + 1)]