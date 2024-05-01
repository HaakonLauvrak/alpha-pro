#The game to be played, either "nim" or "hex".
game = "hex" 

#The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
board_size = 7

#The number of initial stones N in nim and the number of stones that can be removed each turn K
nim_N = 8
nim_K = 4

#Standard MCTS parameters, such as the number of episodes, number of search games per actual move, etc.
num_episodes = 100
num_search_games = 1000
time_limit = 0 #set to 0 for no time limit
c = 1.3
epsilon = 1
replay_buffer_size = 100000

#In the ANET, the learning rate, the number of hidden layers and neurons per layer, along 
#with any of the following activation functions for hidden nodes: linear, sigmoid, tanh, RELU.
learning_rate = 0.005
epochs = 20
batch_size = 128
dimensions_conv = [32] #hex
dimensions_dense = [128, 256, 128] #hex
dimensions_nim = [64, 32, 16, 8] #nim
activation = "relu" #linear, sigmoid, tanh, relu

#The optimizer in the ANET, with (at least) the following options all available: 
#Adagrad, Stochastic Gradient Descent
#  (SGD), RMSProp, and Adam.
optimizer = "adam" #adagrad, sgd, rmsprop, adam

#The number (M) of ANETs to be cached in preparation for a TOPP. 
M = 4
#The number of games (G) to be played between each pair of ANETs in a TOPP.
G = 20

