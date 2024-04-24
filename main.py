from asyncio import sleep
import copy
import random
import threading
import time

import numpy as np
import pygame
from actor.replay_buffer import REPLAY_BUFFER
from game_logic.nim_state_manager import NIM_STATE_MANAGER
from gui.hex_board import HEX_BOARD
from gui.hex_game_gui import HEX_GAME_GUI
from game_logic.hex_state_manager import HEX_STATE_MANAGER
from gui.nim_board import NIM_BOARD
from mcts.mcts import MCTSNode, MonteCarloTreeSearch
from anytree import Node, RenderTree
import config.config as config
from actor.anet import ANET
from tournaments.topp import Tournament
import matplotlib.pyplot as plt
from play.play import PLAY


if __name__ == "__main__":

    PLAY().search_and_train_hex()
    # PLAY().generate_training_data_hex()
    # PLAY().train_hex_actor()
    # PLAY().search_and_train_nim()
    # PLAY().play_hex()
    # PLAY().play_nim()
    # PLAY().topp()


    