import copy
import random

import numpy as np
import config.config as config
from gui.hex_board import HEX_BOARD
from game_logic.state_manager import STATE_MANAGER


class HEX_STATE_MANAGER(STATE_MANAGER):
    def __init__(self, gui) -> None:
        super().__init__()
        self.gui = gui
    
    def getLegalMoves(self, state) -> list:
        if self.isGameOver(state):
            return []
        return [cell.position for cell in state[0].get_cells() if cell.state == 0]
    
    def makeMove(self, move, state) -> None:
        if move not in self.getLegalMoves(state):
            raise ValueError("Invalid move: " + str(move) + " in state: " + str(state[0].get_state(state[1])) + " with legal moves: " + str(self.getLegalMoves(state)))
        state[0].set_cell(move, state[1])
        state[1] = 1 if state[1] == -1 else -1
        self.gui.updateBoard(state[0])

    
    def simulateMove(self, state, actor, move=None) -> tuple:
        """Simulate a move for the given state and actor.
        Args:
            state (tuple): The current state of the game.
            actor (int): The policy network making the move.
            move (tuple, optional): The move to make. Move=none for rollout simulations.
        """
        
        if move is None: 
            move = super().findMove(state, actor)
        
        if move not in self.getLegalMoves(state):
            raise ValueError("Invalid move: " + str(move) + " in state: " + str(state[0].get_state(state[1])) + " with legal moves: " + str(self.getLegalMoves(state)))
        
        new_board = copy.deepcopy(state[0])
        new_board.set_cell(move, state[1])
        new_player_turn = 1 if state[1] == -1 else -1
        return [new_board, new_player_turn]
        
    def isGameOver(self, state) -> bool:
        board, current_player = state
        board_size = config.board_size

        def dfs(cell, player):
            if player == 1 and cell.position[0] == board_size - 1:  # Player 1 reaches the bottom
                super(HEX_STATE_MANAGER, self).increment_episode()
                return True
            if player == -1 and cell.position[1] == board_size - 1:  # Player -1 reaches the right side
                super(HEX_STATE_MANAGER, self).increment_episode()
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
        return -state[1]
    
    def find_all_moves(self):
        all_moves = []
        for i in range(config.board_size):
            for j in range(config.board_size):
                all_moves.append((i, j))
        return all_moves
    
    def getLegalMovesList(self, state):
        legal_moves = self.getLegalMoves(state)
        legal_moves_list = [1 if (i, j) in legal_moves else 0 for i in range(config.board_size) for j in range(config.board_size)]
        return legal_moves_list
