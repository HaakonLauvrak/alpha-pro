from abc import ABC, abstractmethod

class STATE_MANAGER(ABC):
    
    def setState(self, state):
        self.state = state

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
