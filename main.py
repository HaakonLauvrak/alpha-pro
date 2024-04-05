from asyncio import sleep
import copy
import threading
import time

import pygame
from game_logic.nim_state_manager import NIM_STATE_MANAGER
from gui.hex_board import HEX_BOARD
from gui.hex_game_gui import HEX_GAME_GUI
from game_logic.hex_state_manager import HEX_STATE_MANAGER
from gui.nim_board import NIM_BOARD
from mcts.mcts import MCTSNode, MonteCarloTreeSearch
from anytree import Node, RenderTree
import config.config as config
from actor.anet import ANET

def start_gui(game_gui):
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        game_gui.drawBoard()
        pygame.time.delay(100)  # Delay to limit the number of redraws per second
    pygame.quit()
if __name__ == "__main__":

    # NIM GAME TEST ###
    # dict = {}
    # for i in range(100):
    #     state = [[config.nim_N], 1]
    #     sm = NIM_STATE_MANAGER()
    #     ann = ANET()
    #     mcts = MonteCarloTreeSearch(state, ann, sm)
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


    ## HEX GAME MCTS WITH ANN TEST ###

    # game_gui = HEX_GAME_GUI()
    # sm = HEX_STATE_MANAGER(game_gui)
    # anet = ANET()
    # state = [HEX_BOARD(config.board_size), 1]
    # game_gui.updateBoard(state[0])
    # mcts = MonteCarloTreeSearch(state, anet, sm)
    # gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    # gui_thread.start()
    # acc = []
    # for i in range(2):
    #     print(i)
    #     while not sm.isGameOver(state):
    #         mcts.search()
    #         bestAction = mcts.best_action()
    #         sm.makeMove(bestAction, state)
    #         mcts.update_root(bestAction)
    #     training_data = mcts.extract_training_data()
    #     acc.append(anet.train_model(training_data))
    #     state = [HEX_BOARD(config.board_size), 1]
    #     game_gui.updateBoard(state[0])
    #     sm.setState(state)
    #     mcts = MonteCarloTreeSearch(state, anet, sm)
    # print(acc)
    # print("Done")

    # NIM GAME MCTS WITH ANN TEST ###

    sm = NIM_STATE_MANAGER()
    anet = ANET()
    board = NIM_BOARD()
    board.set_state(config.nim_N)
    state = [board, 1]
    mcts = MonteCarloTreeSearch(state, anet, sm)
    acc = []
    for i in range(3):
        print(i)
        while not sm.isGameOver(state):
            mcts.search()
            bestAction = mcts.best_action()
            print("----------------------------------")
            print("State: ", state[0].get_state())
            print("Best action: ", bestAction)
            print("----------------------------------")
            sm.makeMove(bestAction, state)
            mcts.update_root(bestAction)
        training_data = mcts.extract_training_data()
        print("----------------------------------")
        print(training_data)
        print("----------------------------------")
        acc.append(anet.train_model(training_data))
        board = NIM_BOARD()
        board.set_state(config.nim_N)
        state = [board, 1]
        sm.setState(state)
        mcts = MonteCarloTreeSearch(state, anet, sm)
    print(acc)
    print("Done")
