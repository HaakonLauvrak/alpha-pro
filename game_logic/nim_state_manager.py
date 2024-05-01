import copy
import random

import numpy as np

from game_logic.state_manager import STATE_MANAGER
import config.config as config

class NIM_STATE_MANAGER(STATE_MANAGER):
    """
    This class represents the state manager for the game of Nim.
    It inherits from the base class STATE_MANAGER.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def getLegalMoves(self, state):
        """
        Returns a list of legal moves for the given state.
        
        Parameters:
        - state: The current state of the game.
        
        Returns:
        - legal_moves: A list of legal moves.
        """
        return [i for i in range(1, (min(config.nim_K, state[0].get_state())) + 1)]
    
    def makeMove(self, move, state):
        """
        Makes a move in the game by updating the state.
        
        Parameters:
        - move: The move to be made.
        - state: The current state of the game.
        """
        state[0].set_state(state[0].get_state() - move)
        state[1] = 1 if state[1] == -1 else -1
    
    def simulateMove(self, state, actor, move=None, random_move=False):
        """
        Simulates a move in the game by creating a new state.
        
        Parameters:
        - state: The current state of the game.
        - actor: The actor that makes the move.
        - move: The move to be made (optional).
        - random_move: Whether to make a random move (optional).
        
        Returns:
        - new_state: The new state after the move is made.
        """
        if move is None: 
            if random_move:
                move = random.choice(self.getLegalMoves(state))
            else:
                move = self.findMove(state, actor)
        
        if move not in self.getLegalMoves(state):
            raise ValueError("Invalid move: " + str(move) + " in state: " + str(state[0].get_state(state[1])) + " with legal moves: " + str(self.getLegalMoves(state)))

        new_state = copy.deepcopy(state)
        new_state[0].set_state(new_state[0].get_state() - move)
        new_state[1] = 1 if new_state[1] == -1 else -1
        return new_state
    
    def isGameOver(self, state):
        """
        Checks if the game is over.
        
        Parameters:
        - state: The current state of the game.
        
        Returns:
        - game_over: True if the game is over, False otherwise.
        """
        if state[0].get_state() <= 0:
            return True
        return False
    
    def getReward(self, state):
        """
        Returns the reward for the given state.
        
        Parameters:
        - state: The current state of the game.
        
        Returns:
        - reward: The reward for the state.
        """
        return -state[1]
    
    def find_all_moves(self):
        """
        Returns a list of all possible moves in the game.
        
        Returns:
        - all_moves: A list of all possible moves.
        """
        return [i for i in range(1, config.nim_K + 1)]
    
    def getLegalMovesList(self, state):
        """
        Returns a binary list indicating the legality of moves.
        
        Parameters:
        - state: The current state of the game.
        
        Returns:
        - legal_moves: A binary list indicating the legality of moves.
        """
        legal_moves = self.getLegalMoves(state)
        return [1 if i in legal_moves else 0 for i in range(1, config.nim_K + 1)]
    
    def findMove(self, state, actor, greedy=False) -> tuple[int, int]:
        """
        Finds the best move to make based on the current state and actor.
        
        Parameters:
        - state: The current state of the game.
        - actor: The actor that makes the move.
        - greedy: Whether to use greedy strategy (optional).
        
        Returns:
        - move: The best move to make.
        """
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