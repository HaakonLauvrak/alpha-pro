from game_logic.state_manager import STATE_MANAGER
import config.config as config

class NIM_STATE_MANAGER(STATE_MANAGER):

    def __init__(self, player_to_start):
        self.state = self.generateInitialState(player_to_start)

    def getState(self):
        return self.state
    
    def generateInitialState(self, player_to_start):
        return [config.nim_N, player_to_start]
    
    def getLegalMoves(self, state):
        return [i for i in range(1, (min(config.nim_K, state[0])) + 1)]
    
    def makeMove(self, move):
        self.state[0] -= move
        self.state[1] = 1 if self.state[1] == -1 else -1
    
    def simulateMove(self, move, state):
        new_state = state.copy()
        new_state[0] -= move
        new_state[1] = 1 if new_state[1] == -1 else -1
        return new_state
    
    def isGameOver(self, state):
        return state[0] <= 0
    
    def getReward(self, state):
        return -state[1]