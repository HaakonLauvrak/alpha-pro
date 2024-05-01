import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class HEX_BOARD_VISUALIZER():
    """
    A class for visualizing a hex board game using matplotlib and networkx.
    """

    def __init__(self, board_size):
        """
        Initializes the HEX_BOARD_VISUALIZER class.

        Args:
            board_size (int): The size of the hex board.
        """
        self.board_size = board_size
        self.graph = nx.Graph()
        self.pos = {}  # Position dictionary for nodes
        self.fig, self.ax = plt.subplots(dpi=100)
        
        # Initialize the graph and positions
        for r in range(board_size):
            for c in range(board_size):
                node = (r, c)
                # Apply rotation transformation
                x_rot = r * np.cos(np.radians(-135)) - c * np.sin(np.radians(-135))
                y_rot = 1.5 * (r * np.sin(np.radians(-135)) + c * np.cos(np.radians(-135)))
                self.pos[node] = (x_rot, y_rot)
                if r > 0:
                    self.graph.add_edge((r - 1, c), node)
                if c > 0:
                    self.graph.add_edge((r, c - 1), node)
                if r > 0 and c < board_size - 1:
                    self.graph.add_edge((r - 1, c + 1), (r, c))
                if r < board_size - 1 and c > 0:
                    self.graph.add_edge((r + 1, c - 1), (r, c))

        self.nodes = nx.draw_networkx_nodes(self.graph, self.pos, node_color='lightgray', node_size=300)
        nx.draw_networkx_edges(self.graph, self.pos, width=1.5, alpha=0.5)
        self.ax.set_aspect('equal')
        plt.axis('off')
        plt.ion()  # Turn on interactive mode
        plt.show()

    def update_board(self, cells):
        """
        Updates the board visualization based on the state of the cells.

        Args:
            cells (list): The list of cells representing the state of the board.
        """
        red_cell_cords = [cell.position for cell in cells if cell.state == 1]
        blue_cell_cords = [cell.position for cell in cells if cell.state == -1]
        for node in self.graph.nodes():
            if node in red_cell_cords:
                self.graph.nodes[node]['color'] = 'red'
            elif node in blue_cell_cords:
                self.graph.nodes[node]['color'] = 'blue'
            else:
                self.graph.nodes[node]['color'] = 'lightgray'
        self.colors = [self.graph.nodes[node]['color'] for node in self.graph.nodes()]
        self.nodes.set_color(self.colors)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """
        Closes the plot and turns off interactive mode.
        """
        plt.ioff()
        plt.close(self.fig)

    def save(self, filename):
        """
        Saves the plot as an image file.

        Args:
            filename (str): The filename of the image file to be saved.
        """
        self.fig.savefig(filename)