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
import config.config as config
from actor.anet import ANET
from tournaments.topp import Tournament
import matplotlib.pyplot as plt

class PLAY():

    def __init__(self) -> None:
        pass

    def start_gui(self, game_gui):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            game_gui.drawBoard()
            # Delay to limit the number of redraws per second
            pygame.time.delay(100)
        pygame.quit()

    def play_nim_mcts(): 
        """Play a game of NIM using MCTS without ANET"""
        board = NIM_BOARD()
        board.set_state(config.nim_N)
        state = [board, 1]
        sm = NIM_STATE_MANAGER()
        ann = ANET()
        print("Starting nim game")
        print(f"N = {config.nim_N}, K = {config.nim_K}")
        mcts = MonteCarloTreeSearch(state, ann, sm)
        while not sm.isGameOver(state):
            mcts.search()
            bestAction = mcts.best_action()
            sm.makeMove(bestAction, state)
            print(f"Player {state[1]} took: ", bestAction)
        print(f"Player {state[1] * -1} wins!")
    
    def play_hex_mcts(self):
        """Play a game of HEX using MCTS without ANET"""
        game_gui = HEX_GAME_GUI()
        sm = HEX_STATE_MANAGER(game_gui)
        anet = ANET("training_net")
        state = [HEX_BOARD(config.board_size), 1]
        gui_thread = threading.Thread(target=self.start_gui, args=(game_gui,))
        gui_thread.start()
        while not sm.isGameOver(state):
            mcts = MonteCarloTreeSearch(state, anet, sm)
            bestAction = mcts.search()
            sm.makeMove(bestAction, state)
            print(f"Player {state[1]} took: ", bestAction)
        print(f"Player {state[1] * -1} wins!")
        self.running = False
        gui_thread.join()

    def search_and_train_hex(self):
        """Play a game of HEX using MCTS with ANET"""
        game_gui = HEX_GAME_GUI()
        sm = HEX_STATE_MANAGER(game_gui)
        anet = ANET("training_net")
        state = [HEX_BOARD(config.board_size), 1]
        game_gui.updateBoard(state[0])
        mcts = MonteCarloTreeSearch(state, anet, sm)
        gui_thread = threading.Thread(target=self.start_gui, args=(game_gui,))
        gui_thread.start()
        acc = []
        loss = []
        mae = []
        replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
        for i in range(config.num_episodes):
            print(i)
            while not sm.isGameOver(state):
                mcts.search()
                bestAction = mcts.best_action()
                print(bestAction)
                sm.makeMove(bestAction, state)
                mcts.update_root(bestAction)
            sm.increment_episode() #must be done here to avoid incrementing episode when simulations reach end state.
            x_train, y_train = mcts.extract_training_data()
            replay_buffer.add(x_train, y_train)
            training_score = (anet.train_model(replay_buffer.get_all()))
            loss.append(training_score[0])
            acc.append(training_score[1])
            mae.append(training_score[2])
            state = [HEX_BOARD(config.board_size), 1 if i % 2 == 0 else -1]
            game_gui.updateBoard(state[0])
            sm.setState(state)
            mcts = MonteCarloTreeSearch(state, anet, sm)
            if i % (config.num_episodes // config.M) == 0:
                anet.save_model(i + 1)
        self.running = False
        gui_thread.join()
        print(acc)

        fig = plt.figure()

        # Add subplots
        ax1 = fig.add_subplot(311)
        ax1.plot(acc, label='Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='upper left')

        ax2 = fig.add_subplot(312)
        ax2.plot(loss, label='Loss', color='orange')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper left')

        ax3 = fig.add_subplot(313)
        ax3.plot(mae, label='Mean Absolute Error', color='green')
        ax3.set_title('Mean Absolute Error')
        ax3.set_ylabel('MAE')
        ax3.set_xlabel('Games')
        ax3.legend(loc='upper left')

        plt.tight_layout()

        # Show the figure
        plt.show()

        print("Done")
    

    def search_and_train_nim(self):
        sm = NIM_STATE_MANAGER()
        anet = ANET("nim")
        board = NIM_BOARD()
        board.set_state(config.nim_N)
        state = [board, 1]
        mcts = MonteCarloTreeSearch(state, anet, sm)
        acc = []
        for i in range(config.num_episodes):
            print(i)
            while not sm.isGameOver(state):
                mcts.search()
                bestAction = mcts.best_action()
                sm.makeMove(bestAction, state)
                mcts.update_root(bestAction)
            sm.increment_episode()

            training_data = mcts.extract_training_data()
            acc.append(anet.train_model(training_data))
            board = NIM_BOARD()
            board.set_state(config.nim_N)
            state = [board, 1]
            sm.setState(state)
            mcts = MonteCarloTreeSearch(state, anet, sm)
        print(acc)
        print("Done")

    def topp(self, models: list):
        """Takes a list of four models and plays a tournament between them"""
        anet0 = ANET("Player0")
        anet1 = ANET("Player1")
        anet2 = ANET("Player2")
        anet3 = ANET("Player3")

        it0_model = anet0.load_model(models[0])
        anet0.set_model(it0_model)
        it1_model = anet1.load_model(models[1])
        anet1.set_model(it1_model)
        it2_model = anet2.load_model(models[2])
        anet2.set_model(it2_model)
        it3__model = anet3.load_model(models[3])
        anet3.set_model(it3__model)
        

        players = [anet0, anet1, anet2, anet3]
        gui = HEX_GAME_GUI()
        sm = HEX_STATE_MANAGER(gui)
        tournament = Tournament(players, sm, 50, "hex")
        results = tournament.play_tournament()
        main_dict = {}
        for dict in results:
            for key in dict:
                if key in main_dict:
                    main_dict[key] += dict[key]
                else:
                    main_dict[key] = dict[key]
            print(dict)
        print(main_dict)

        #plot main dict as bar chart
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(main_dict.keys(), main_dict.values())
        plt.show()

    def generate_training_data_hex(self):
        """Generates training data for HEX using MCTS without anet. Saves data to file."""
        game_gui = HEX_GAME_GUI()
        sm = HEX_STATE_MANAGER(game_gui)
        anet = ANET("training_net")
        state = [HEX_BOARD(config.board_size), 1]
        game_gui.updateBoard(state[0])
        gui_thread = threading.Thread(target=self.start_gui, args=(game_gui,))
        gui_thread.start()
        replay_buffer = REPLAY_BUFFER(10000)
        for i in range(10):
            if replay_buffer.get_size() == 10000:
                break
            print(i)
            if i % 2 == 0:
                state = [HEX_BOARD(config.board_size), -1]
            else:
                state = [HEX_BOARD(config.board_size), 1]
            mcts = MonteCarloTreeSearch(state, anet, sm)
            while not sm.isGameOver(state):
                mcts.search(random=True)
                bestAction = mcts.best_action()
                sm.makeMove(bestAction, state)
                mcts.update_root(bestAction)

            x_train, y_train = mcts.extract_training_data()
            replay_buffer.add(x_train, y_train)
            print(replay_buffer.get_size())
        replay_buffer.save("training_data/hex_training_data")

    def train_hex_actor(self):
        replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
        replay_buffer.load("training_data/hex_training_data.npz")
        
        acc = []
        loss = []
        mae = []
    
        for i in range(1):
            print(i)
            anet = ANET("training_net")
            training_score = anet.train_model(replay_buffer.get_all())
            loss.append(training_score[0])
            acc.append(training_score[1])
            mae.append(training_score[2])
        anet.save_model(10)

        # Print avg accuracy, loss and mae
        print("Avg accuracy: ", sum(acc) / len(acc))
        print("Avg loss: ", sum(loss) / len(loss))
        print("Avg mae: ", sum(mae) / len(mae))
        print("Learning rate: ", config.learning_rate)
        print("Epochs: ", config.epochs)
        print("Batch size: ", config.batch_size)
        print("Dimensions conv: ", config.dimensions_conv)
        print("Dimensions dense: ", config.dimensions_dense)
        print("Activation: ", config.activation)
        print("Optimizer: ", config.optimizer)

        #Save results to file
        with open("training_results_hex.txt", "a") as f:
            f.write("Avg accuracy: " + str(sum(acc) / len(acc)) + "\n")
            f.write("Avg loss: " + str(sum(loss) / len(loss)) + "\n")
            f.write("Avg mae: " + str(sum(mae) / len(mae)) + "\n")
            f.write("Learning rate: " + str(config.learning_rate) + "\n")
            f.write("Epochs: " + str(config.epochs) + "\n")
            f.write("Batch size: " + str(config.batch_size) + "\n")
            f.write("Dimensions conv: " + str(config.dimensions_conv) + "\n")
            f.write("Dimensions dense: " + str(config.dimensions_dense) + "\n")
            f.write("Activation: " + str(config.activation) + "\n")
            f.write("Optimizer: " + str(config.optimizer) + "\n" + "\n")
