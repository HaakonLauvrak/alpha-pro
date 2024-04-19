from asyncio import sleep
import copy
import threading
import time

import numpy as np
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
from tournament.tournament import Tournament
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

    ## HEX GAME GUI TEST ###
    # game_gui = HEX_GAME_GUI()
    # sm = HEX_STATE_MANAGER(game_gui)
    # anet = ANET("training_net")
    # state = [HEX_BOARD(config.board_size), 1]
    # gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    # game_gui.updateBoard(state[0])
    # gui_thread.start()
    # sm.makeMove((0, 0), state)
    # sm.makeMove((1, 2), state)
    # sm.makeMove((1, 0), state)
    # sm.makeMove((2, 2), state)
    # sm.makeMove((2, 0), state)
    # mcts = MonteCarloTreeSearch(state, anet, sm)
    # mcts.search()
    # bestAction = mcts.best_action()
    


    # HEX GAME MCTS WITH ANN TEST ###

    game_gui = HEX_GAME_GUI()
    sm = HEX_STATE_MANAGER(game_gui)
    anet = ANET("training_net")
    state = [HEX_BOARD(config.board_size), 1]
    game_gui.updateBoard(state[0])
    mcts = MonteCarloTreeSearch(state, anet, sm)
    gui_thread = threading.Thread(target=start_gui, args=(game_gui,))
    gui_thread.start()
    acc = []
    loss = []
    mae = []
    training_data = {"x_train": np.array([]), "y_train": np.array([])}
    for i in range(config.num_episodes):
        print(i)
        while not sm.isGameOver(state):
            mcts.search()
            bestAction = mcts.best_action()
            print(bestAction)
            sm.makeMove(bestAction, state)
            mcts.update_root(bestAction)
        training_data_for_game = mcts.extract_training_data()
        if training_data["x_train"].size == 0:
            training_data["x_train"] = training_data_for_game["x_train"]
            training_data["y_train"] = training_data_for_game["y_train"]
        else:
            training_data["x_train"] = np.concatenate((training_data["x_train"], training_data_for_game["x_train"]))
            training_data["y_train"] = np.concatenate((training_data["y_train"], training_data_for_game["y_train"]))
        training_score = (anet.train_model(training_data))
        acc.append(training_score[0])
        loss.append(training_score[1])
        mae.append(training_score[2])
        state = [HEX_BOARD(config.board_size), 1 if i % 2 == 0 else -1]
        game_gui.updateBoard(state[0])
        sm.setState(state)
        mcts = MonteCarloTreeSearch(state, anet, sm)
        if (i + 1) % 5 == 0 or i == 0: 
            anet.save_model(i + 1)
    # LAGRE DATA FOR FLERE GAMES OG TRENE 
    # Sjekke om mcts knuser random
    print(acc)
    #Plot the accuracy
    fig = plt.figure()

    #Add subplots
    #Subplot 1: Accuracy
    ax1 = fig.add_subplot(311) 
    ax1.plot(acc, label='Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')

    # Subplot 2: Loss
    ax2 = fig.add_subplot(312) 
    ax2.plot(loss, label='Loss', color='orange')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper left')

    # Subplot 3: MAE
    ax3 = fig.add_subplot(313) 
    ax3.plot(mae, label='Mean Absolute Error', color='green')
    ax3.set_title('Mean Absolute Error')
    ax3.set_ylabel('MAE')
    ax3.set_xlabel('Games')
    ax3.legend(loc='upper left')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the figure
    plt.show()
    
    print("Done")

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

    # it0_model = anet0.load_model("hex", 5, 5000)
    # anet0.set_model(it0_model)
    # it1_model = anet1.load_model("hex", 10, 5000)
    # anet1.set_model(it1_model)
    # it2_model = anet2.load_model("hex", 60, 20000)
    # anet2.set_model(it2_model)
    # it3__model = anet3.load_model("hex", 80, 20000)
    # anet3.set_model(it3__model)
    # it4__model = anet4.load_model("hex", 100, 20000)
    # anet4.set_model(it4__model)
    # it5__model = anet5.load_model("hex", 120, 20000)
    # anet5.set_model(it5__model)
    # it6__model = anet6.load_model("hex", 140, 20000)


    # players = [anet0, anet1]
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

    # plot main dict as bar chart
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.bar(main_dict.keys(), main_dict.values())
    # plt.show()


# [0.010746645741164684, 0.005752491299062967, 0.00914386473596096, 0.006938515696674585, 0.029249554499983788,
#  0.019246697425842285, 0.08489418029785156, 0.05262318626046181, 0.093918576836586, 0.10453776270151138,
#  0.09627887606620789, 0.10363857448101044, 0.10118476301431656, 0.09498618543148041, 0.10507889837026596,
#  0.10231374949216843, 0.09827683120965958, 0.10385973751544952, 0.10182685405015945, 0.10247606039047241]