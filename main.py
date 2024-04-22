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

    ### NIM GAME TEST ###
    # dict = {}
    # for i in range(100):
    #     print(i)
    #     board = NIM_BOARD()
    #     board.set_state(config.nim_N)
    #     state = [board, 1]
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
    # for i in range(1):
    #     print(i+1)
    #     game_gui = HEX_GAME_GUI()
    #     sm = HEX_STATE_MANAGER(game_gui)
    #     anet = ANET("training_net")
    #     state = [HEX_BOARD(config.board_size), 1]
    #     gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    #     gui_thread.start()
    #     sm.makeMove((0, 0), state)
    #     sm.makeMove((1, 2), state)
    #     sm.makeMove((1, 0), state)
    #     sm.makeMove((2, 2), state)
    #     sm.makeMove((2, 0), state)
    #     mcts = MonteCarloTreeSearch(state, anet, sm)
    #     bestAction = mcts.search()
    #     if bestAction in dict:
    #         dict[bestAction] += 1
    #     else:
    #         dict[bestAction] = 1
    # print(dict)


    # HEX GAME MCTS WITH ANN TEST ###

    # game_gui = HEX_GAME_GUI()
    # sm = HEX_STATE_MANAGER(game_gui)
    # anet = ANET("training_net")
    # state = [HEX_BOARD(config.board_size), 1]
    # game_gui.updateBoard(state[0])
    # mcts = MonteCarloTreeSearch(state, anet, sm)
    # gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    # gui_thread.start()
    # acc = []
    # loss = []
    # mae = []
    # replay_buffer = REPLAY_BUFFER(10000)
    # for i in range(config.num_episodes):
    #     print(i)
    #     while not sm.isGameOver(state):
    #         mcts.search()
    #         bestAction = mcts.best_action()
    #         print(bestAction)
    #         sm.makeMove(bestAction, state)
    #         mcts.update_root(bestAction)
    #     x_train, y_train = mcts.extract_training_data()
    #     replay_buffer.add(x_train, y_train)
    #     training_score = (anet.train_model(replay_buffer.get_data()))
    #     acc.append(training_score[0])
    #     loss.append(training_score[1])
    #     mae.append(training_score[2])
    #     state = [HEX_BOARD(config.board_size), 1 if i % 2 == 0 else -1]
    #     game_gui.updateBoard(state[0])
    #     sm.setState(state)
    #     mcts = MonteCarloTreeSearch(state, anet, sm)
    #     if (i + 1) % 5 == 0 or i == 0: 
    #         anet.save_model(i + 1)

    
    # print(acc)
    # #Plot the accuracy
    # fig = plt.figure()

    # #Add subplots
    # #Subplot 1: Accuracy
    # ax1 = fig.add_subplot(311) 
    # ax1.plot(acc, label='Accuracy')
    # ax1.set_title('Model Accuracy')
    # ax1.set_ylabel('Accuracy')
    # ax1.legend(loc='upper left')

    # # Subplot 2: Loss
    # ax2 = fig.add_subplot(312) 
    # ax2.plot(loss, label='Loss', color='orange')
    # ax2.set_title('Model Loss')
    # ax2.set_ylabel('Loss')
    # ax2.legend(loc='upper left')

    # # Subplot 3: MAE
    # ax3 = fig.add_subplot(313) 
    # ax3.plot(mae, label='Mean Absolute Error', color='green')
    # ax3.set_title('Mean Absolute Error')
    # ax3.set_ylabel('MAE')
    # ax3.set_xlabel('Games')
    # ax3.legend(loc='upper left')

    # # Adjust layout to prevent overlap
    # plt.tight_layout()

    # # Show the figure
    # plt.show()
    
    # print("Done")

    # # NIM GAME MCTS WITH ANN TEST ###

    # sm = NIM_STATE_MANAGER()
    # anet = ANET("nimminimminim")
    # board = NIM_BOARD()
    # board.set_state(config.nim_N)
    # state = [board, 1]
    # mcts = MonteCarloTreeSearch(state, anet, sm)
    # acc = []
    # for i in range(100):
    #     print(i)
    #     while not sm.isGameOver(state):
    #         mcts.search()
    #         bestAction = mcts.best_action()
    #         print("----------------------------------")
    #         print("State: ", state[0].get_state())
    #         print("Best action: ", bestAction)
    #         print("----------------------------------")
    #         sm.makeMove(bestAction, state)
    #         mcts.update_root(bestAction)
    #     training_data = mcts.extract_training_data()
    #     print("----------------------------------")
    #     print(training_data)
    #     print("----------------------------------")
    #     acc.append(anet.train_model(training_data))
    #     board = NIM_BOARD()
    #     board.set_state(config.nim_N)
    #     state = [board, 1]
    #     sm.setState(state)
    #     mcts = MonteCarloTreeSearch(state, anet, sm)
    # print(acc)
    # print("Done")

    ## HEX TURNAMENT TEST ##
    # anet0 = ANET("Player0")
    # anet1 = ANET("Player1")
    # anet2 = ANET("Player2")
    # anet3 = ANET("Player3")
    # anet4 = ANET("Player4")
    # anet5 = ANET("Player5")
    # anet6 = ANET("Player6")

    # it0_model = anet0.load_model("hex", 7,  5, 15000)
    # anet0.set_model(it0_model)
    # it1_model = anet1.load_model("hex", 7,  10, 15000)
    # anet1.set_model(it1_model)
    # it2_model = anet2.load_model("hex", 7,  60, 15000)
    # anet2.set_model(it2_model)
    # it3__model = anet3.load_model("hex", 7,  80, 15000)
    # anet3.set_model(it3__model)
    # it4__model = anet4.load_model("hex", 7,  100, 15000)
    # anet4.set_model(it4__model)
    # it5__model = anet5.load_model("hex", 7,  120, 15000)
    # anet5.set_model(it5__model)
    # it6__model = anet6.load_model("hex", 7,  150, 15000)


    # players = [anet0, anet2, anet4, anet6]
    # gui = HEX_GAME_GUI()
    # sm = HEX_STATE_MANAGER(gui)
    # tournament = Tournament(players, sm, 20, "hex")
    # results = tournament.play_tournament()
    # main_dict = {}
    # for dict in results:
    #     for key in dict:
    #         if key in main_dict:
    #             main_dict[key] += dict[key]
    #         else:
    #             main_dict[key] = dict[key]
    #     print(dict)
    # print(main_dict)

    # #plot main dict as bar chart
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.bar(main_dict.keys(), main_dict.values())
    # plt.show()


    # GENERATE TRAINING DATA FOR HEX ##
    game_gui = HEX_GAME_GUI()
    sm = HEX_STATE_MANAGER(game_gui)
    anet = ANET("training_net")
    state = [HEX_BOARD(config.board_size), 1]
    game_gui.updateBoard(state[0])
    gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    gui_thread.start()
    replay_buffer = REPLAY_BUFFER(10000)
    for i in range(10):
        print(i)
        if i % 2 == 0:
            state = [HEX_BOARD(config.board_size), -1]
        else:
            state = [HEX_BOARD(config.board_size), 1]
        mcts = MonteCarloTreeSearch(state, anet, sm)
        while not sm.isGameOver(state):
            mcts.search()
            bestAction = mcts.best_action()
            print(bestAction)
            sm.makeMove(bestAction, state)
            mcts.update_root(bestAction)
        x_train, y_train = mcts.extract_training_data()
        replay_buffer.add(x_train, y_train)
        print(replay_buffer.get_size())
    replay_buffer.save("training_data/hex_training_data")

    ## TRAIN NEURAL NET ON HEX TRAINING DATA ##

    # replay_buffer = REPLAY_BUFFER(10000)
    # replay_buffer.load("training_data/hex_training_data.npz")
    # acc = []
    # loss = []
    # mae = []
    # for i in range(5):
    #     print(i)
    #     anet = ANET("training_net")
    #     training_score = anet.train_model(replay_buffer.get_all())
    #     acc.append(training_score[0])
    #     loss.append(training_score[1])
    #     mae.append(training_score[2])
    
    ## Print avg accuracy, loss and mae
    # print("Avg accuracy: ", sum(acc) / len(acc))
    # print("Avg loss: ", sum(loss) / len(loss))
    # print("Avg mae: ", sum(mae) / len(mae))