from abc import ABC, abstractmethod
import random
import config.config as config
import numpy as np

class STATE_MANAGER(ABC):
    
    def __init__(self) -> None:
        self.epsilon = config.epsilon
        self.current_episode = 0
        self.num_episodes = config.num_episodes

    def setState(self, state) -> None:
        self.state = state
    
    def increment_episode(self):
        self.current_episode += 1

    @abstractmethod
    def getLegalMoves(self, state):
        pass

    # Modify your state based on the move and change the player to move next
    @abstractmethod
    def makeMove(self, move):
        pass

    @abstractmethod
    def simulateMove(self, move, state):
        pass

    @abstractmethod
    def isGameOver(self):
        pass

    @abstractmethod
    def find_all_moves(self):
        pass

    @abstractmethod
    def findMove(self, state, actor):
        pass

    