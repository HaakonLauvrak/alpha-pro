import copy
import config.config as config
from gui.hex_board import HEX_BOARD
from game_logic.state_manager import STATE_MANAGER


class HEX_STATE_MANAGER(STATE_MANAGER):
    def __init__(self, player_to_start, gui):
        self.state = [HEX_BOARD(config.board_size), player_to_start]
        self.gui = gui

    def getState(self):
        return self.state
    
    def getLegalMoves(self, state) -> list:
        if self.isGameOver(state):
            return []
        return [cell.position for cell in state[0].get_cells() if cell.state == 0]
    
    def makeMove(self, move) -> None:
        if move not in self.getLegalMoves(self.state):
            raise ValueError("Invalid move")
        self.state[0].set_cell(move, self.state[1])
        self.state[1] = 1 if self.state[1] == -1 else -1
        self.gui.updateBoard(self.state[0])
    
    def simulateMove(self, move, state) -> tuple:
        if move not in self.getLegalMoves(state):
            raise ValueError("Invalid move")
        
        new_board = HEX_BOARD(config.board_size)

        for cell in state[0].get_cells():
            new_board.set_cell(cell.position, cell.state)
        new_board.set_cell(move, state[1])
        new_player_turn = -state[1]
        return [new_board, new_player_turn]
        
        
    def isGameOver(self, state) -> bool:
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
        return -state[1]


