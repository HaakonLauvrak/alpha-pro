import copy
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
                results.append(self.play(player1, player2))
        return results
    

    def make_new_game(self):
        if self.game_type == "hex":
            return [HEX_BOARD(config.board_size), 1]
        

    def play(self, player1, player2):
        results = {player1: 0, player2: 0}
        for i in range(self.rounds):
            current_state = self.make_new_game()
            all_moves = self.state_manager.find_all_moves()
            while not self.state_manager.isGameOver(current_state):
                if current_state[1] == 1:
                    move_probabilities = player1.compute_move_probabilities(current_state)[0]
                    legal_moves = self.state_manager.getLegalMovesList(current_state)
                    for i in range(len(legal_moves)):
                        if legal_moves[i] == 0:
                            move_probabilities[i] = 0
                    best_move_index = np.argmax(move_probabilities)
                    move = all_moves[best_move_index]
                else:
                    move_probabilities = player2.compute_move_probabilities(current_state)[0]
                    legal_moves = self.state_manager.getLegalMovesList(current_state)
                    for i in range(len(legal_moves)):
                        if legal_moves[i] == 0:
                            move_probabilities[i] = 0
                    best_move_index = np.argmax(move_probabilities)
                    move = all_moves[best_move_index]
                self.state_manager.makeMove(move, current_state)
            if self.state_manager.getReward(current_state) == -1:
                results[player1] += 1
            else:
                results[player2] += 1
        return results
        


