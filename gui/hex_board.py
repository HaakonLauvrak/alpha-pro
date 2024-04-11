import numpy as np
from gui.board import BOARD
import config.config as config

class HEX_BOARD(BOARD):
    
    def __init__(self, board_size) -> None:
        self.cells = []
        for i in range(board_size):
            for j in range(board_size):
                self.cells.append(HEX_CELL((i, j), self))

        for cell in self.cells: 
            cell.set_neighbours()
            

    def get_cells(self) -> list:
        return self.cells
    
    def set_cell(self, position, player):

        for cell in self.cells:
            if cell.position == position:
                cell.state = player
                break
     
    def setCells(self, cells):
        self.cells = cells


    def __str__(self) -> str:
        return str([cell.state for cell in self.cells])
    
    def get_state(self, player):
        list_format = [cell.state for cell in self.cells] + [player]
        list_format = np.array(list_format)
        list_format = np.expand_dims(list_format, axis=0)
        return list_format
    
    def get_ann_input(self, player):
        cell_states = [cell.state for cell in self.cells]
        ann_input = np.array([np.array([np.array([0, 0]) for i in range(config.board_size)]) for i in range(config.board_size)])
        for i in range(len(cell_states)):
            ann_input[(i//config.board_size)] = np.array([cell_states[i], player])
        ann_input = np.expand_dims(ann_input, axis=0)
        return ann_input
    
class HEX_CELL():
    
    def __init__(self, position, board) -> None:
        self.state = 0
        self.position = position
        self.board = board
        self.neighbours = []
        
    def set_neighbours(self) -> None:
        cells = self.board.get_cells()
        for cell in cells:
            if cell.position[0] == self.position[0] + 1 and cell.position[1] == self.position[1]:
                self.neighbours.append(cell)
            elif cell.position[0] == self.position[0] - 1 and cell.position[1] == self.position[1]:
                self.neighbours.append(cell)
            elif cell.position[0] == self.position[0] and cell.position[1] == self.position[1] + 1:
                self.neighbours.append(cell)
            elif cell.position[0] == self.position[0] and cell.position[1] == self.position[1] - 1:
                self.neighbours.append(cell)
            elif cell.position[0] == self.position[0] + 1 and cell.position[1] == self.position[1] - 1:
                self.neighbours.append(cell)
            elif cell.position[0] == self.position[0] - 1 and cell.position[1] == self.position[1] + 1:
                self.neighbours.append(cell)
    
    def __eq__(self, other):
        if not isinstance(other, HEX_CELL):
            return False
        return self.position == other.position and self.state == other.state

    def __hash__(self):
        return hash((self.position, self.state))


