import math
import random
import numpy as np
from anytree import Node, RenderTree
from game_logic.nim_state_manager import NIM_STATE_MANAGER
import config.config as config


class MCTSNode(Node):
    def __init__(self, state, parent=None, action=None):
        name = str(state)
        super().__init__(name, parent=parent)
        self.state = state
        self.action = action
        self.win_score_1 = 0
        self.win_score_2 = 0
        self.draw = 0
        self.visits = 0
        self.anet_probabilities = []


class MonteCarloTreeSearch:
    def __init__(self, root_state, actor_network, state_manager):
        self.root = MCTSNode(root_state)
        self.original_root = self.root
        self.actor_network = actor_network
        self.state_manager = state_manager
        self.all_moves = self.find_all_moves()

    def find_all_moves(self):
        all_moves = []
        for i in range(config.board_size):
            for j in range(config.board_size):
                all_moves.append((i, j))
        return all_moves

    def select_node(self) -> MCTSNode:
        node = self.root
        while node.children:
            node.visits += 1
            node = self.tree_policy(node)
        return node

    def expand_node(self, node):
        node.visits += 1
        legal_moves = self.state_manager.getLegalMoves(node.state)
        for move in legal_moves:
            new_state = self.state_manager.simulateMove(move, node.state)
            MCTSNode(new_state, parent=node, action=move)

    def rollout(self, node):
        # Simulate a game from the current state
        current_state = node.state
        # while not self.state_manager.isGameOver(current_state):
        while not self.state_manager.isGameOver(current_state):
            legal_moves = self.state_manager.getLegalMoves(current_state)
            # legal_moves_list = []
            # for i in range(config.board_size):
            #     for j in range(config.board_size):
            #         if (i, j) in legal_moves:
            #             legal_moves_list.append(1)
            #         else:
            #             legal_moves_list.append(0)

            # probabilites = self.actor_network.compute_move_probabilities(current_state)[
            #     0]
            # node.anet_probabilities = probabilites
            # probabilites_normalized = [
            #     probabilites[i] if legal_moves_list[i] == 1 else 0 for i in range(len(legal_moves_list))]
            # if sum(probabilites_normalized) == 0:
            #     move = random.choice(legal_moves)
            # else:
            #     probabilites_normalized = [
            #     x / sum(probabilites_normalized) for x in probabilites_normalized]
            #     move = random.choices(population = self.all_moves, weights = probabilites_normalized)[0]
            move = random.choice(legal_moves)
            current_state = self.state_manager.simulateMove(
                move, current_state)
        self.backpropagate(node, self.state_manager.getReward(current_state))

    def backpropagate(self, node, reward):
        # Update the nodes in the path to the root with the reward
        while node:
            if reward == 1:
                node.win_score_1 += 1
            elif reward == -1:
                node.win_score_2 += 1
            else:
                node.draw += 1
            node = node.parent

    def best_action(self):
        return sorted(self.root.children, key=lambda c: c.visits, reverse=True)[0].action

    def tree_policy(self, node):
        # Return the best child node according to the UCT policy
        max_score = -np.inf
        best_child = []
        for child in node.children:
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
            return 0
        else:
            if player == 1:
                return (a.win_score_1 - a.win_score_2) / a.visits

            else:
                return (a.win_score_2 - a.win_score_1) / a.visits

    @staticmethod
    def u(s, a):
        # Return the UCT exploration value for a given state and action
        return 0.2*np.sqrt(math.log((s.visits) / (1 + a.visits)))

    def search(self):
        for _ in range(100):
            node = self.select_node()
            self.expand_node(node)
            self.rollout(node)
        return self.best_action()

    def update_root(self, move):
        for child in self.root.children:
            if child.action == move:
                self.root = child
                self.root.parent = None
                move_found = True
                break
        if not move_found:
            raise ValueError("Invalid move")

    def extract_training_data(self):
        training_data = {"x_train": np.array([]), "y_train": np.array([])}
        node_queue = [self.original_root]
        # Traverse the tree
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            if node.children:
                visits = {}
                for move in self.all_moves:
                    visits[move] = 0
                for child in node.children:
                    visits[child.action] = child.visits
                    if child.children:
                        node_queue.append(child)
                visits_list = np.array(list(visits.values()))
                sum_visits = sum(visits_list)
                if sum_visits > 0:
                    visits_list = [x / sum_visits for x in visits_list]
                    print(visits_list)
                    exit()
                    training_data["x_train"] = np.append(training_data["x_train"],
                                                        node.state[0].get_cells_as_list(node.state[1])[0])
                    training_data["y_train"] = np.append(
                        training_data["y_train"], visits_list)
        return training_data
