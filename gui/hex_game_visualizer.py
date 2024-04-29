import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class HEX_BOARD_VISUALIZER():
    def __init__(self, board_size):
        self.board_size = board_size
        self.graph = nx.Graph()
        self.pos = {}  # Position dictionary for nodes
        self.fig, self.ax = plt.subplots(dpi=100)
        
        # Initialize the graph and positions
        for r in range(board_size):
            for c in range(board_size):
                node = (r, c)
                index = r * board_size + c
                x, y = index // board_size, index % board_size
                # Apply rotation transformation
                x_rot = x * np.cos(np.radians(-45)) - y * np.sin(np.radians(-45))
                y_rot = 1.5 * (x * np.sin(np.radians(-45)) + y * np.cos(np.radians(-45)))
                self.pos[node] = (x_rot, y_rot)
                if r > 0:
                    self.graph.add_edge((r - 1, c), node)
                if c > 0:
                    self.graph.add_edge((r, c - 1), node)
                if r > 0 and c > 0:
                    self.graph.add_edge((r - 1, c - 1), (r, c))

        self.nodes = nx.draw_networkx_nodes(self.graph, self.pos, node_color='lightgray', node_size=300)
        nx.draw_networkx_edges(self.graph, self.pos, width=1.5, alpha=0.5)
        self.ax.set_aspect('equal')
        plt.axis('off')
        plt.ion()  # Turn on interactive mode
        plt.show()

    def update_board(self, board_state):
        node_colors = ['red' if x == 1 else 'blue' if x == -1 else 'lightgray' for x in board_state]
        self.nodes.set_color(node_colors)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()  # Turn off interactive mode
        plt.close(self.fig)


    def save(self, filename):
        self.fig.savefig(filename)