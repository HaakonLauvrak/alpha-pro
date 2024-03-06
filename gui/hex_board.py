class HEX_BOARD():
    
    def __init__(self, board_size) -> None:
        self.cells = []
        for i in range(board_size):
            for j in range(board_size):
                self.cells.append(HEX_CELL((i, j), self))
        
        for cell in self.cells: 
            cell.set_neighbours()

    def get_cells(self) -> list:
        return self.cells

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


