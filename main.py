from game_logic.nim_state_manager import NIM_STATE_MANAGER


game = NIM_STATE_MANAGER(1)
print(game.getState())
print(game.getLegalMoves())
game.makeMove(3)
print(game.getState())