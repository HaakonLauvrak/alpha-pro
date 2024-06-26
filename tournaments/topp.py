import config.config as config
from gui.hex_board import HEX_BOARD
from gui.hex_game_visualizer import HEX_BOARD_VISUALIZER

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
            visualizer = HEX_BOARD_VISUALIZER(config.board_size)
            for i in range(self.rounds):
                print("Round: " + str(i + 1))
                current_state = self.make_new_game()
                visualizer.update_board(current_state[0].get_cells())
                while not self.state_manager.isGameOver(current_state):
                    if current_state[1] == 1:
                        move = self.state_manager.findMove(current_state, player1)
                    else:
                        move = self.state_manager.findMove(current_state, player2)
                    self.state_manager.makeMove(move, current_state)
                    visualizer.update_board(current_state[0].get_cells())
                if current_state[1] == -1:
                    results[player1.name] += 1
                else: 
                    results[player2.name] += 1
            visualizer.close()
            return results


            


