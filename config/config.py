#The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
board_size = 4

#The number of initial stones N in nim and the number of stones that can be removed each turn K
nim_N = 7
nim_K = 5

#Standard MCTS parameters, such as the number of episodes, number of search games per actual move, etc.
num_episodes = 100
num_search_games = 100

#In the ANET, the learning rate, the number of hidden layers and neurons per layer, along 
#with any of the following activation functions for hidden nodes: linear, sigmoid, tanh, RELU.
learning_rate = 0.01
num_hidden_layers = 2
layers = [board_size for _ in range(num_hidden_layers)]
activation = "relu"

#The optimizer in the ANET, with (at least) the following options all available: 
#Adagrad, Stochastic Gradient Descent (SGD), RMSProp, and Adam.
optimizer = "adagrad"

#The number (M) of ANETs to be cached in preparation for a TOPP. These should be cached, starting with an untrained net prior to episode 1, at a fixed interval throughout the training episodes.
M = 4

#The number of games, G, to be played between any two ANET-based agents that meet during the round-robin play of the TOPP.
G = 10