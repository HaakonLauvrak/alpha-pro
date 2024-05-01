import numpy as np
from gui.board import BOARD
import config.config as config

class HEX_BOARD(BOARD):
    """
    Represents a hexagonal game board.
    """
    
    def __init__(self, board_size) -> None:
        """
        Initializes the HEX_BOARD object.

        Args:
            board_size (int): The size of the board.
        """
        self.cells = []

        for i in range(board_size):
            for j in range(board_size):
                self.cells.append(HEX_CELL((i, j), self))

        for cell in self.cells: 
            cell.set_neighbours()
            

    def get_cells(self) -> list:
        """
        Returns the list of cells on the board.
        """
        return self.cells
    
    def set_cell(self, position, player):
        """
        Sets the state of a cell at a given position to a given player.

        Args:
            position (tuple): The position of the cell.
            player (int): The player number.
        """
        for cell in self.cells:
            if cell.position == position:
                cell.set_state(player)
                break
     
    def setCells(self, cells):
        """
        Sets the list of cells on the board.

        Args:
            cells (list): A list of HEX_CELL objects representing the cells on the board.
        """
        self.cells = cells


    def __str__(self) -> str:
        """
        Returns a string representation of the board.
        """
        return str([cell.state for cell in self.cells])
    
    def get_state(self, player):
        """
        Returns the state of the board as a numpy array.

        Args:
            player (int): The player number.
        """
        list_format = [cell.state for cell in self.cells] + [player]
        list_format = np.array(list_format)
        list_format = np.expand_dims(list_format, axis=0)
        return list_format
    
    def get_state_list(self):
        """
        Returns the state of the board as a list.
        """
        return [cell.state for cell in self.cells]
    
    def get_ann_input(self, player):
        """
        Returns the state of the board as a numpy array suitable for input to an artificial neural network.

        Args:
            player (int): The player number.
        """
        cell_states = [cell.state for cell in self.cells]
        ann_input = np.array([np.array([np.array([0, 0, 0]) for i in range(config.board_size)]) for i in range(config.board_size)])
        for i in range(len(cell_states)):
            if cell_states[i] == 1:
                ann_input[(i//config.board_size)][(i%config.board_size)] = np.array([1, 0, player])
            elif cell_states[i] == -1:
                ann_input[(i//config.board_size)][(i%config.board_size)] = np.array([0, 1, player])
            else:
                ann_input[(i//config.board_size)][(i%config.board_size)] = np.array([0, 0, player])
        ann_input = np.expand_dims(ann_input, axis=0)
        return ann_input
    
class HEX_CELL():
    """
    Represents a cell on a hexagonal game board.
    """

    def __init__(self, position, board) -> None:
        """
        Initializes the HEX_CELL object.

        Args:
            position (tuple): The position of the cell.
            board (HEX_BOARD): The HEX_BOARD object that the cell belongs to.
        """
        self.state = 0
        self.position = position
        self.board = board
        self.neighbours = []

    def __str__(self) -> str:
        """
        Returns a string representation of the cell.
        """
        return "Pos: " + str(self.position) + " State: " + str(self.state)
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the cell.
        """
        return "Pos: " + str(self.position) + " State: " + str(self.state)
    
    def set_state(self, state) -> None:
        """
        Sets the state of the cell.

        Args:
            state (int): The state of the cell.
        """
        self.state = state

    def set_neighbours(self) -> None:
        """
        Sets the neighbouring cells.
        """
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


