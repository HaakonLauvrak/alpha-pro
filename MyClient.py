from game_logic.hex_state_manager import HEX_STATE_MANAGER
from tournaments import ActorClient
from gui.hex_board import HEX_BOARD
import config.config as config
import random 
from actor.anet import ANET 


class MyClient(ActorClient.ActorClient):
    def __init__(self):
        super().__init__()

    def handle_get_action(self, state):
        # Implement your own logic here
        self.board = HEX_BOARD(7)
        self.actor = ANET("oht")
        self.actor.load_model("hex", 7, 150, 15000)

        state = state[1:]
        for i in range(len(state)):
            if state[i] == 2:     
                self.board.set_cell((i // config.board_size, i % config.board_size), -1)
            elif state[i] == 1:
                self.board.set_cell((i // config.board_size, i% config.board_size), 1)
        current_state = [self.board, 1]
        
        self.state_manager = HEX_STATE_MANAGER(self.board)    
        
        return self.state_manager.findMove(current_state, self.actor, greedy=True)
    
if __name__ == "__main__":
    client = MyClient()
    client.run()

