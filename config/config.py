#The game to be played, either "nim" or "hex".
game = "hex" 

#The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
board_size = 7

#The number of initial stones N in nim and the number of stones that can be removed each turn K
nim_N = 8
nim_K = 4

#Standard MCTS parameters, such as the number of episodes, number of search games per actual move, etc.
num_episodes = 150
num_search_games = 15000
time_limit = 10
c = 1
epsilon = 0.1

#In the ANET, the learning rate, the number of hidden layers and neurons per layer, along 
#with any of the following activation functions for hidden nodes: linear, sigmoid, tanh, RELU.
learning_rate = 0.005
epochs = 10
batch_size = 64
dimensions_conv = [32, 32, 16] #hex
dimensions_dense = [64, 64, 32, 16] #hex
# dimensions = [64, 32, 16, 8] #nim
activation = "relu"

#The optimizer in the ANET, with (at least) the following options all available: 
#Adagrad, Stochastic Gradient Descent
#  (SGD), RMSProp, and Adam.
optimizer = "adam"

#The number (M) of ANETs to be cached in preparation for a TOPP. These should be cached, starting with an untrained net prior to episode 1, at a fixed interval throughout the training episodes.
M = 4

#The number of games, G, to be played between any two ANET-based agents that meet during the round-robin play of the TOPP.
G = 5