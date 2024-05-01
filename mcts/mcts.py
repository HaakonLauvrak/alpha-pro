import copy
import math
import pprint
import random
import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DictExporter
import tensorflow as tf
from game_logic.nim_state_manager import NIM_STATE_MANAGER
import config.config as config
import time


class MCTSNode(Node):
    def __init__(self, state, parent=None, action=None, visits=0):
        # name = str(state[0])
        super().__init__(".", parent=parent)
        self.state = state
        self.action = action
        self.win_score_1 = 0
        self.win_score_2 = 0
        self.draw = 0
        self.visits = visits

class MonteCarloTreeSearch:
    def __init__(self, root_state, actor_network, state_manager):
        self.root = MCTSNode(copy.deepcopy(root_state), visits=1) #Set root visits to 1 to avoid log of zero
        self.actor_network = actor_network
        self.state_manager = state_manager
        self.all_moves = self.state_manager.find_all_moves()

    def select_node(self) -> MCTSNode:
        node = self.root
        while node.children:
            node = self.tree_policy(node)
        return node

    def expand_node(self, node):
        legal_moves = self.state_manager.getLegalMoves(node.state)
        for move in legal_moves:
            new_state = self.state_manager.simulateMove(node.state, self.actor_network, move=move)
            MCTSNode(new_state, parent=node, action=move)

    def rollout(self, node, random_move=False):
        # Simulate a game from the current state
        current_state = node.state
        while not self.state_manager.isGameOver(current_state):
            current_state = self.state_manager.simulateMove(current_state, self.actor_network, random_move=random_move)
        self.backpropagate(node, self.state_manager.getReward(current_state))

    def backpropagate(self, node, reward):
        # Update the nodes in the path to the root with the reward
        while node:
            node.visits += 1
            if reward == 1:
                node.win_score_1 += 1
            else:
                node.win_score_2 += 1
            if node is self.root:
                break
            node = node.parent

    def best_action(self):
        return sorted(self.root.children, key=lambda c: c.visits, reverse=True)[0].action 

    def tree_policy(self, node):
        # Return the best child node according to the UCT policy
        max_score = -np.inf
        best_child = []
        for child in node.children:
            if child.visits == 0:
                return child
            score = MonteCarloTreeSearch.Q(
                child, node.state[1]) + MonteCarloTreeSearch.u(node, child)
            if score == max_score:
                best_child.append(child)
            elif score > max_score:
                max_score = score
                best_child = [child]
        return random.choice(best_child)

    @staticmethod
    def Q(a, player):
        # Return the action value for a given state and action
        if a.visits == 0:
            return np.inf
        else:
            if player == 1:
                return (a.win_score_1 - a.win_score_2) / a.visits
            else:
                return (a.win_score_2 - a.win_score_1) / a.visits

    @staticmethod
    def u(s, a):
        # Return the UCT exploration value for a given state and action
        if (s.visits) / (1 + a.visits) <= 1:
            return 0
        return config.c * np.sqrt(math.log((s.visits) / (1 + a.visits)))
 
    def search(self, random_move=False):
        time_limit = config.time_limit
        start_time = time.time()
        for i in range(config.num_search_games):
            if config.time_limit > 0 and time.time() - start_time > time_limit:
                 break
            node = self.select_node()
            if not self.state_manager.isGameOver(node.state):  
                self.expand_node(node)
                node = random.choice(node.children)
            self.rollout(node, random_move=random_move)
        return self.best_action()

    def update_root(self, move):
        move_found = False
        for child in self.root.children:
            if child.action == move:
                self.root = child
                move_found = True
                break
        if not move_found:
            raise ValueError(f"Invalid move: {move}")

    def extract_training_data(self):
        original_root = self.root
        while self.root.parent:
            self.root = self.root.parent
        if self.root.children:
            visits = {}
            for move in self.all_moves:
                visits[move] = 0
            for child in self.root.children:
                visits[child.action] = child.visits
            visits_list = np.array(list(visits.values()))
            sum_visits = sum(visits_list)
            visits_list = [x / sum_visits for x in visits_list]
            x_train = self.root.state[0].get_ann_input(self.root.state[1])[0]
        self.root = original_root
        print(x_train)
        print(visits_list)
        exit()
        return x_train, visits_list


    # def extract_training_data(self):
    #     original_root = self.root
    #     while self.root.parent:
    #         self.root = self.root.parent
    #     training_data = {"x_train": [], "y_train": []}
    #     node_queue = [self.root]
    #     # Traverse the tree
    #     while len(node_queue) > 0:
    #         node = node_queue.pop(0)
    #         if node.children:
    #             visits = {}
    #             for move in self.all_moves:
    #                 visits[move] = 0
    #             for child in node.children:
    #                 visits[child.action] = child.visits
    #                 if child.children:
    #                     node_queue.append(child)
    #             visits_list = np.array(list(visits.values()))
    #             sum_visits = sum(visits_list)
    #             if (sum_visits > 0):
    #                 visits_list = [x / sum_visits for x in visits_list]
    #                 x_train = node.state[0].get_ann_input(node.state[1])
    #                 training_data["x_train"].append(x_train[0])
    #                 training_data["y_train"].append(visits_list)
    #     self.root = original_root
    #     return training_data["x_train"], training_data["y_train"]


