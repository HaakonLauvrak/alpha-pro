import math
import sys
import pygame

from gui.hex_board import HEX_BOARD


class HEX_GAME_GUI():
    def __init__(self, board_size):
        pygame.init()
        self.board_size = board_size
        self.cell_radius = 30
        self.screen_width = 3 * self.cell_radius * board_size
        self.screen_height = int(1.75 * self.cell_radius * board_size)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.board = HEX_BOARD(board_size)
        self.colors = {0: (255, 255, 255), 1: (255, 0, 0), 2: (0, 0, 255)}
        
    def draw_hex(self, x, y, color):
        # Now drawing hexagons instead of triangles
        points = [(math.cos(2 * math.pi / 6 * i) * self.cell_radius + x,
                   math.sin(2 * math.pi / 6 * i) * self.cell_radius + y)
                  for i in range(6)]
        pygame.draw.polygon(self.screen, color, points)

    def draw_board(self):
        self.screen.fill((0, 0, 0)) 

        dx = self.cell_radius * 3**0.5
        dy = self.cell_radius * 1.5
        for cell in self.board.get_cells():
            # Calculate the offset based on cell position
            x_offset = dx * (cell.position[1] + cell.position[0] / 2)
            y_offset = dy * cell.position[0]

            # Draw the hexagon
            hex_center_x = x_offset + self.screen_width / 2 - dx * self.board_size / 2
            hex_center_y = y_offset + self.cell_radius
            self.draw_hex(hex_center_x, hex_center_y, self.colors[cell.state])

            # If the cell is not empty, draw a circle to represent the player's move
            if cell.state != 0:
                pygame.draw.circle(self.screen, self.colors[cell.state], (int(hex_center_x), int(hex_center_y)), self.cell_radius // 2)

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.draw_board()
            clock.tick(60)
        pygame.quit()
        sys.exit()

