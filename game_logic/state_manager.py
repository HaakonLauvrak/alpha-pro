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

    def findMove(self, state, actor, greedy=False) -> tuple[int, int]:
        self.epsilon =  1 - self.current_episode / config.num_episodes
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

    