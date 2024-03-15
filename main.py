from asyncio import sleep
import copy
import threading
import time
from game_logic.nim_state_manager import NIM_STATE_MANAGER
from gui.hex_board import HEX_BOARD
from gui.hex_game_gui import HEX_GAME_GUI
from game_logic.hex_state_manager import HEX_STATE_MANAGER
from mcts.mcts import MCTSNode, MonteCarloTreeSearch
from anytree import Node, RenderTree
import config.config as config
from actor.anet import ANET

def start_gui(game_gui):
    game_gui.run()

if __name__ == "__main__":

    ### NIM GAME TEST ###
    # dict = {}
    # for i in range(100):
    #     sm = NIM_STATE_MANAGER(1)
    #     mcts = MonteCarloTreeSearch(sm.getState(), None, sm)
    #     mcts.search()
    #     bestAction = mcts.best_action()
    #     if bestAction in dict:
    #         dict[bestAction] += 1
    #     else:
    #         dict[bestAction] = 1
    # print(dict)

    ### HEX GAME TEST ###

    # dict = {}
    # for i in range(100):
    #     game_gui = HEX_GAME_GUI()
    #     sim_gui = HEX_GAME_GUI()
    #     sm = HEX_STATE_MANAGER(1, game_gui)
    #     sim_sm = HEX_STATE_MANAGER(1, sim_gui)
    #     game_gui.updateBoard(sm.getState()[0])
    #     # gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    #     # gui_thread.start()
    #     sm.makeMove((0, 0))
    #     sm.makeMove((2, 2))
    #     sm.makeMove((2, 0))
    #     sm.makeMove((1, 0))
    #     mcts = MonteCarloTreeSearch(sm.getState(), None, sim_sm)
    #     game_gui.updateBoard(sm.getState()[0])
    #     mcts.search()
    #     game_gui.updateBoard(sm.getState()[0])
    #     bestAction = mcts.best_action()
    #     if bestAction in dict:
    #         dict[bestAction] += 1
    #     else:
    #         dict[bestAction] = 1
    # print(dict)

    ## HEX GAME GUI TEST ###
    # game_gui = HEX_GAME_GUI()
    # sm = HEX_STATE_MANAGER(1, game_gui)
    # game_gui.updateBoard(sm.getState()[0])
    # gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    # gui_thread.start()
    # sm.makeMove((0, 0))
    # sm.makeMove((1, 2))
    # sm.makeMove((1, 0))
    # sm.makeMove((2, 2))
    # print(sm.isGameOver(sm.getState()))
    # sm.makeMove((2, 0))
    # print(sm.isGameOver(sm.getState()))

    game_gui = HEX_GAME_GUI()
    sm = HEX_STATE_MANAGER(game_gui)
    board = HEX_BOARD(config.board_size)
    state = [board, 1]
    game_gui.updateBoard(state[0])
    anet = ANET()
    mcts = MonteCarloTreeSearch(state, anet, sm)
    gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    gui_thread.start()
    for i in range(3):
        while not sm.isGameOver(state):
            mcts.search()
            bestAction = mcts.best_action()
            print(bestAction)
            sm.makeMove(bestAction, state)
            mcts.update_root(bestAction)
            print(state)
        training_data = mcts.extract_training_data()
        print(training_data)
        print(anet.train_model(training_data))
        print("Game Over")

