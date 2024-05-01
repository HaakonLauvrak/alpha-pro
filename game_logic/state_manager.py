from abc import ABC, abstractmethod
import config.config as config

class STATE_MANAGER(ABC):
    """
    Abstract base class for managing the game state.
    """

    def __init__(self, epsilon=config.epsilon) -> None:
        """
        Initializes the STATE_MANAGER object.
        """
        self.epsilon = epsilon
        self.current_episode = 0
        self.num_episodes = config.num_episodes

    def setState(self, state) -> None:
        """
        Sets the current state of the game.

        Parameters:
        - state: The current state of the game.
        """
        self.state = state
    
    def increment_episode(self):
        """
        Increments the episode counter.
        """
        self.current_episode += 1

    @abstractmethod
    def getLegalMoves(self, state):
        pass

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

    