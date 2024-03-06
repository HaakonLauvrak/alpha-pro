from game_logic.nim_state_manager import NIM_STATE_MANAGER
from gui.hex_game_gui import HEX_GAME_GUI
from mcts.mcts import MonteCarloTreeSearch

if __name__ == "__main__":
    game_gui = HEX_GAME_GUI(6)  # Initialize with desired board size
    game_gui.run()