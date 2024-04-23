import copy
import random

import numpy as np

from game_logic.state_manager import STATE_MANAGER
import config.config as config

class NIM_STATE_MANAGER(STATE_MANAGER):

    def __init__(self) -> None:
        super().__init__()
    
    def getLegalMoves(self, state):
        return [i for i in range(1, (min(config.nim_K, state[0].get_state())) + 1)]
    
    def makeMove(self, move, state):
        state[0].set_state(state[0].get_state() - move)
        state[1] = 1 if state[1] == -1 else -1
    
    def simulateMove(self, state, actor):
        move = self.findMove(state, actor)
        new_state = copy.deepcopy(state)
        new_state[0].set_state(new_state[0].get_state() - move)
        new_state[1] = 1 if new_state[1] == -1 else -1
        return new_state
    
    def isGameOver(self, state):
        if state[0].get_state() <= 0:
            return True
        return False
    
    def getReward(self, state):
        return -state[1]
    
    def find_all_moves(self):
        return [i for i in range(1, config.nim_K + 1)]
    
    def getLegalMovesList(self, state):
        legal_moves = self.getLegalMoves(state)
        return [1 if i in legal_moves else 0 for i in range(1, config.nim_K + 1)]
    
    def findMove(self, state, actor, greedy=False) -> tuple[int, int]:
        self.epsilon = 1 - self.current_episode / config.num_episodes
        all_moves = self.find_all_moves()
        probabilities = actor.compute_move_probabilities(state[0].get_ann_input(state[1]))[0]
        legal_moves = self.getLegalMovesList(state)
        probabilites_normalized = [probabilities[i] if legal_moves[i] == 1 else 0 for i in range(len(legal_moves))]
        probabilites_normalized = [x / sum(probabilites_normalized) for x in probabilites_normalized]

        if sum(probabilities) == 0:
            move = random.choice(self.getLegalMoves(state))
        else:
            if greedy: 
                greedy_index = np.argmax(probabilites_normalized)
                move = all_moves[greedy_index]
            else: 
                if self.epsilon > random.random():
                    move = random.choice(self.getLegalMoves(state))
                else: 
                    move = random.choices(population = all_moves, weights = probabilites_normalized)[0]
        return move