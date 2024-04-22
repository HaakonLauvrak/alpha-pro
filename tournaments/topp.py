import copy
import random
import numpy as np
import config.config as config
from gui.hex_board import HEX_BOARD

class Tournament():

    def __init__(self, players, state_manager, rounds, game_type):
        self.players = players
        self.state_manager = state_manager
        self.rounds = rounds
        self.game_type = game_type

    def play_tournament(self):
        results = []
        for player1 in self.players:
            for player2 in self.players:
                if player1 == player2:
                    continue
                print("Playing: " + player1.name + " vs " + player2.name)
                results.append(self.play(player1, player2))
        return results
    

    def make_new_game(self):
        if self.game_type == "hex":
            return [HEX_BOARD(config.board_size), 1]
        
def play(self, player1, player2):
        results = {player1.name: 0, player2.name: 0}
        for i in range(self.rounds):
            print("Round: " + str(i + 1))
            current_state = self.make_new_game()
            while not self.state_manager.isGameOver(current_state):
                if current_state[1] == 1:
                    move = self.state_manger.findMove(current_state, player1)
                else:
                    move = self.state_manger.findMove(current_state, player2)
                self.state_manager.makeMove(move, current_state)
            if self.state_manager.getReward(current_state) == -1:
                results[player1.name] += 1
            else:
                results[player2.name] += 1
        return results

    # def play(self, player1, player2):
    #     results = {player1.name: 0, player2.name: 0}
    #     for i in range(self.rounds):
    #         print("Round: " + str(i + 1))
    #         current_state = self.make_new_game()
    #         all_moves = self.state_manager.find_all_moves()
    #         while not self.state_manager.isGameOver(current_state):
    #             if current_state[1] == 1:
    #                 probabilities = player1.compute_move_probabilities(current_state[0].get_ann_input(current_state[1]))[0]
    #                 if sum(probabilities) == 0:
    #                     move = random.choice(self.state_manager.getLegalMoves(current_state))
    #                 else:
    #                     legal_moves = self.state_manager.getLegalMovesList(current_state)
    #                     probabilites_normalized = [probabilities[i] if legal_moves[i] == 1 else 0 for i in range(len(legal_moves))]
    #                     probabilites_normalized = [x / sum(probabilites_normalized) for x in probabilites_normalized]
    #                     move = random.choices(population = all_moves, weights = probabilites_normalized)[0]
    #             else:
    #                 probabilities = player2.compute_move_probabilities(current_state[0].get_ann_input(current_state[1]))[0]
    #                 if sum(probabilities) == 0:
    #                     move = random.choice(self.state_manager.getLegalMoves(current_state))
    #                 else:
    #                     legal_moves = self.state_manager.getLegalMovesList(current_state)
    #                     probabilites_normalized = [probabilities[i] if legal_moves[i] == 1 else 0 for i in range(len(legal_moves))]
    #                     probabilites_normalized = [x / sum(probabilites_normalized) for x in probabilites_normalized]
    #                     move = random.choices(population = all_moves, weights = probabilites_normalized)[0]
                    
    #             self.state_manager.makeMove(move, current_state)
    #         if self.state_manager.getReward(current_state) == -1:
    #             results[player1.name] += 1
    #         else:
    #             results[player2.name] += 1
    #     return results


        


