from asyncio import sleep
import copy
import os
import random
import threading
import time

import numpy as np
from actor.replay_buffer import REPLAY_BUFFER
from game_logic.nim_state_manager import NIM_STATE_MANAGER
from gui.hex_board import HEX_BOARD
from gui.hex_game_visualizer import HEX_BOARD_VISUALIZER
from game_logic.hex_state_manager import HEX_STATE_MANAGER
from gui.nim_board import NIM_BOARD
from mcts.mcts import MCTSNode, MonteCarloTreeSearch
import config.config as config
from actor.anet import ANET
from tournaments.topp import Tournament
import matplotlib.pyplot as plt
import training_data

class PLAY():

    def __init__(self) -> None:
        pass

    def play_nim_mcts(self): 
        """Play a game of NIM using MCTS without ANET"""
        board = NIM_BOARD()
        board.set_state(config.nim_N)
        state = [board, 1]
        sm = NIM_STATE_MANAGER()
        ann = ANET("nim")
        print("Starting nim game")
        print(f"N = {config.nim_N}, K = {config.nim_K}")
        mcts = MonteCarloTreeSearch(state, ann, sm)
        while not sm.isGameOver(state):
            bestAction = mcts.search(random_move=True)
            sm.makeMove(bestAction, state)
            print(f"Player {-state[1]} took: ", bestAction)
        print(f"Player {-state[1]} wins!")


    def play_hex_mcts(self):
        """Play a game of HEX using MCTS without ANET"""
        sm = HEX_STATE_MANAGER()
        anet = ANET("training_net")
        state = [HEX_BOARD(config.board_size), 1]
        mcts = MonteCarloTreeSearch(state, anet, sm)
        #FOR TESTING
        replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
        visualizer = HEX_BOARD_VISUALIZER(config.board_size)
        visualizer.update_board(state[0].get_cells())
        while not sm.isGameOver(state):
            bestAction = mcts.search(random_move=True)
            sm.makeMove(bestAction, state)
            x_train, y_train = mcts.extract_training_data()
            replay_buffer.add(x_train, y_train)
            mcts.update_root(bestAction)
            visualizer.update_board(state[0].get_cells())
            print(f"Player {state[1]} took: ", bestAction)
        print(f"Player {state[1] * -1} wins!")
        visualizer.close()

    def search_and_train_hex(self):
        """Play a game of HEX using MCTS with ANET"""
        sm = HEX_STATE_MANAGER()
        anet = ANET("training_net")
        state = [HEX_BOARD(config.board_size), 1]
        mcts = MonteCarloTreeSearch(state, anet, sm)
        acc = []
        loss = []
        mae = []
        visualizer = HEX_BOARD_VISUALIZER(config.board_size)
        visualizer.update_board(state[0].get_cells())
        replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
        for i in range(config.num_episodes):
            print(i)
            while not sm.isGameOver(state):
                bestAction = mcts.search()
                print(bestAction)
                sm.makeMove(bestAction, state)
                x_train, y_train = mcts.extract_training_data()
                replay_buffer.add(x_train, y_train)
                mcts.update_root(bestAction)
                visualizer.update_board(state[0].get_cells())
            sm.increment_episode() #must be done here to avoid incrementing episode when simulations reach end state.
            training_score = (anet.train_model(replay_buffer.get_all()))
            loss.append(training_score[0])
            acc.append(training_score[1])
            mae.append(training_score[2])
            state = [HEX_BOARD(config.board_size), 1 if i % 2 == 0 else -1]
            sm.setState(state)
            mcts = MonteCarloTreeSearch(state, anet, sm)
            if config.num_episodes // config.M == 0:
                if i == 0:
                    anet.save_model()
            elif i % (config.num_episodes // config.M) == 0:
                anet.save_model(i + 1)
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

    def topp(self, models: list, rounds):
        """Takes a list of four models and plays a tournament between them"""

        players = []
        anet = ANET("Player0")
        players.append(anet)
        for i in range(len(models)):
            anet = ANET(f"Player{i + 1}")
            model = anet.load_model_by_name(models[i])
            anet.set_model(model)
            players.append(anet)

        sm = HEX_STATE_MANAGER()
        tournament = Tournament(players, sm, rounds, "hex")
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
        sm = HEX_STATE_MANAGER()
        anet = ANET("training_net")
        state = [HEX_BOARD(config.board_size), 1]
        replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
    
        game_counter = 0
        for i in range(config.num_episodes):
            if replay_buffer.get_size() == config.replay_buffer_size:
                replay_buffer.save(f"training_data/hex_training_data_{game_counter}games")
                replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
            if i % 2 == 0:
                state = [HEX_BOARD(config.board_size), -1]
            else:
                state = [HEX_BOARD(config.board_size), 1]
            
            mcts = MonteCarloTreeSearch(state, anet, sm)
            while not sm.isGameOver(state):
                mcts.search(random_move=True)
                bestAction = mcts.best_action()
                sm.makeMove(bestAction, state)
                mcts.update_root(bestAction)
            
            game_counter += 1
            print(f"Game {game_counter} finished")
            x_train, y_train = mcts.extract_training_data()
            replay_buffer.add(x_train, y_train)
            print(f"Replay buffer size: {replay_buffer.get_size()}")
        replay_buffer.save(f"training_data/hex_training_data_{game_counter}games_AA")

    def train_hex_actor(self):
        replay_buffer = REPLAY_BUFFER(10000000)
        replay_buffer.load(f"training_data/hex_100games_10000rollouts.npz")
        print(replay_buffer.get_size())
        
        acc = []
        loss = []
        mae = []

        anet = ANET("training_net")
        training_score = anet.train_model(replay_buffer.get_all())
        loss.append(training_score[0])
        acc.append(training_score[1])
        mae.append(training_score[2])

        anet.save_model(config.num_episodes)

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

            