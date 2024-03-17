import math
import sys
import pygame
import config.config as config

from gui.hex_board import HEX_BOARD


class HEX_GAME_GUI():
    def __init__(self):
        pygame.init()
        self.board_size = config.board_size
        self.cell_radius = 30
        self.screen_width = 3 * self.cell_radius * config.board_size
        self.screen_height = int(1.75 * self.cell_radius * config.board_size)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.colors = {0: (255, 255, 255), 1: (255, 0, 0), -1: (0, 255, 0)}

    def draw_hex(self, x, y, color):
        points = [(x + math.cos(2 * math.pi / 6 * i + math.pi / 6) * self.cell_radius,
                y + math.sin(2 * math.pi / 6 * i + math.pi / 6) * self.cell_radius) for i in range(6)]
        pygame.draw.polygon(self.screen, color, points)

    def updateBoard(self, board):
        self.board = board

    def drawBoard(self):
        self.screen.fill((0, 0, 0))  # Fill the screen with black

        dx = self.cell_radius * 3**0.5
        dy = self.cell_radius * 1.5
        for cell in self.board.get_cells():
            x_offset = dx * (cell.position[1] + cell.position[0] / 2)
            y_offset = dy * cell.position[0]

            hex_center_x = x_offset + self.screen_width / 2 - dx * self.board_size / 2
            hex_center_y = y_offset + self.cell_radius

            # Check if the cell is empty
            if cell.state == 0:
                # Draw an outlined hexagon for an empty cell
                points = [(math.cos(2 * math.pi / 6 * i + math.pi / 6) * self.cell_radius + hex_center_x,
                        math.sin(2 * math.pi / 6 * i + math.pi / 6) * self.cell_radius + hex_center_y) for i in range(6)]
                pygame.draw.lines(self.screen, self.colors[cell.state], True, points, 1)
            else:
                # Draw a filled hexagon for a non-empty cell
                self.draw_hex(hex_center_x, hex_center_y, self.colors[cell.state])

                # If the cell is not empty, draw a circle to represent the player's move
                pygame.draw.circle(self.screen, self.colors[cell.state], (int(hex_center_x), int(hex_center_y)), self.cell_radius // 2)

        pygame.display.flip()



    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.drawBoard()
            clock.tick(60)
        pygame.quit()
        sys.exit()
