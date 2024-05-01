import copy
import random

import numpy as np
import config.config as config
from gui.hex_board import HEX_BOARD
from game_logic.state_manager import STATE_MANAGER


class HEX_STATE_MANAGER(STATE_MANAGER):
    """
    Manages the state and logic of a hex game.
    """

    def __init__(self) -> None:
        super().__init__()

    def getLegalMoves(self, state) -> list:
        """
        Get a list of legal moves for the current state.

        Args:
            state: The current state of the game.

        Returns:
            A list of legal moves.
        """
        if self.isGameOver(state):
            return []
        return [cell.position for cell in state[0].get_cells() if cell.state == 0]

    def makeMove(self, move, state) -> None:
        """
        Make a move in the game.

        Args:
            move: The move to be made.
            state: The current state of the game.

        Raises:
            ValueError: If the move is invalid.
        """
        if move not in self.getLegalMoves(state):
            raise ValueError("Invalid move: " + str(move) + " in state: " + str(state[0].get_state(state[1])) + " with legal moves: " + str(self.getLegalMoves(state)))
        state[0].set_cell(move, state[1])
        state[1] = 1 if state[1] == -1 else -1

    def simulateMove(self, state, actor, move=None, random_move=False) -> tuple:
        """
        Simulate a move in the game.

        Args:
            state: The current state of the game.
            actor: The actor that makes the move.
            move: The move to be simulated. If None, a move will be found using the actor.
            random_move: Whether to choose a random move.

        Raises:
            ValueError: If the move is invalid.

        Returns:
            A tuple containing the new state after the move and the new player turn.
        """
        if move is None:
            move = self.findMove(state, actor, random_move=random_move)

        if move not in self.getLegalMoves(state):
            raise ValueError("Invalid move: " + str(move) + " in state: " + str(state[0].get_state(state[1])) + " with legal moves: " + str(self.getLegalMoves(state)))

        new_board = copy.deepcopy(state[0])
        new_board.set_cell(move, state[1])
        new_player_turn = 1 if state[1] == -1 else -1
        return [new_board, new_player_turn]

    def isGameOver(self, state) -> bool:
        """
        Check if the game is over.

        Args:
            state: The current state of the game.

        Returns:
            True if the game is over, False otherwise.
        """
        board, current_player = state
        board_size = config.board_size

        def dfs(cell, player):
            if player == 1 and cell.position[0] == board_size - 1:  # Player 1 reaches the bottom
                return True
            if player == -1 and cell.position[1] == board_size - 1:  # Player -1 reaches the right side
                return True

            visited.add(cell)
            for neighbour in cell.neighbours:
                if neighbour not in visited and neighbour.state == player:
                    if dfs(neighbour, player):
                        return True
            return False

        # Check for player 1
        visited = set()
        start_cells_p1 = [cell for cell in board.get_cells() if cell.position[0] == 0 and cell.state == 1]
        for cell in start_cells_p1:
            if dfs(cell, 1):
                return True

        # Check for player -1
        visited = set()
        start_cells_p2 = [cell for cell in board.get_cells() if cell.position[1] == 0 and cell.state == -1]
        for cell in start_cells_p2:
            if dfs(cell, -1):
                return True

        return False

    def getReward(self, state):
        """
        Get the reward for the current state.

        Args:
            state: The current state of the game.

        Returns:
            The reward.
        """
        return -state[1]

    def find_all_moves(self):
        """
        Find all possible moves in the game.

        Returns:
            A list of all possible moves.
        """
        all_moves = []
        for i in range(config.board_size):
            for j in range(config.board_size):
                all_moves.append((i, j))
        return all_moves

    def getLegalMovesList(self, state):
        """
        Get a list of legal moves as a binary list.

        Args:
            state: The current state of the game.

        Returns:
            A binary list representing the legal moves.
        """
        legal_moves = self.getLegalMoves(state)
        legal_moves_list = [1 if (i, j) in legal_moves else 0 for i in range(config.board_size) for j in range(config.board_size)]
        return legal_moves_list

    def findMove(self, state, actor, greedy=False, random_move=False) -> tuple[int, int]:
        """
        Find a move to be made in the game.

        Args:
            state: The current state of the game.
            actor: The actor that makes the move.
            greedy: Whether to use greedy strategy.
            random_move: Whether to choose a random move.

        Returns:
            The move to be made.
        """
        if random_move:
            return random.choice(self.getLegalMoves(state))

        self.epsilon = 1 - self.current_episode / config.num_episodes

        if not greedy and self.epsilon > random.random():
            return random.choice(self.getLegalMoves(state))

        all_moves = self.find_all_moves()
        legal_moves = self.getLegalMovesList(state)

        probabilities = actor.compute_move_probabilities(state[0].get_ann_input(state[1]))[0]
        probabilities = [probabilities[i] if legal_moves[i] == 1 else 0 for i in range(config.board_size**2)]

        if sum(probabilities) == 0:
            return random.choice(self.getLegalMoves(state))
        else:
            probabilites_normalized = [x / sum(probabilities) for x in probabilities]
            if greedy:
                greedy_index = np.argmax(probabilites_normalized)
                return all_moves[greedy_index]
            else:
                return random.choices(population=all_moves, weights=probabilites_normalized)[0]
    
    
