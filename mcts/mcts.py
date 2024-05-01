import copy
import math
import random
import numpy as np
import config.config as config
import time
from anytree import Node


class MCTSNode(Node):
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.
    """

    def __init__(self, state, parent=None, action=None, visits=0):
        """
        Initializes a new instance of the MCTSNode class.

        Args:
            state: The state of the node.
            parent: The parent node of this node.
            action: The action taken to reach this node.
            visits: The number of times this node has been visited.
        """
        super().__init__(".", parent=parent)
        self.state = state
        self.action = action
        self.win_score_1 = 0
        self.win_score_2 = 0
        self.draw = 0
        self.visits = visits


class MonteCarloTreeSearch:
    """
    Represents the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes:
        root: The root node of the search tree.
        actor_network: The actor network used for move selection.
        state_manager: The state manager for the game.
        all_moves: A list of all possible moves in the game.
    """

    def __init__(self, root_state, actor_network, state_manager):
        """
        Initializes a new instance of the MonteCarloTreeSearch class.

        Args:
            root_state: The initial state of the game.
            actor_network: The actor network used for move selection.
            state_manager: The state manager for the game.
        """
        self.root = MCTSNode(copy.deepcopy(root_state), visits=1)
        self.actor_network = actor_network
        self.state_manager = state_manager
        self.all_moves = self.state_manager.find_all_moves()

    def select_node(self) -> MCTSNode:
        """
        Selects the next node to explore in the search tree.

        Returns:
            The selected node.
        """
        node = self.root
        while node.children:
            node = self.tree_policy(node)
        return node

    def expand_node(self, node):
        """
        Expands a node by adding child nodes for all legal moves.

        Args:
            node: The node to expand.
        """
        legal_moves = self.state_manager.getLegalMoves(node.state)
        for move in legal_moves:
            new_state = self.state_manager.simulateMove(node.state, self.actor_network, move=move)
            MCTSNode(new_state, parent=node, action=move)

    def rollout(self, node, random_move=False):
        """
        Simulates a game from the current state until a terminal state is reached.

        Args:
            node: The node to start the rollout from.
            random_move: Whether to select moves randomly during the rollout.
        """
        current_state = node.state
        while not self.state_manager.isGameOver(current_state):
            current_state = self.state_manager.simulateMove(current_state, self.actor_network, random_move=random_move)
        self.backpropagate(node, self.state_manager.getReward(current_state))

    def backpropagate(self, node, reward):
        """
        Updates the nodes in the path to the root with the reward.

        Args:
            node: The node to start the backpropagation from.
            reward: The reward obtained from the terminal state.
        """
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
        """
        Returns the best action to take based on the search results.

        Returns:
            The best action to take.
        """
        return sorted(self.root.children, key=lambda c: c.visits, reverse=True)[0].action

    def tree_policy(self, node):
        """
        Selects the best child node according to the UCT policy.

        Args:
            node: The node to select the best child from.

        Returns:
            The best child node.
        """
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
        """
        Returns the action value for a given state and action.

        Args:
            a: The node representing the state and action.
            player: The player for whom to calculate the action value.
        """
        if a.visits == 0:
            return np.inf
        else:
            if player == 1:
                return (a.win_score_1 - a.win_score_2) / a.visits
            else:
                return (a.win_score_2 - a.win_score_1) / a.visits

    @staticmethod
    def u(s, a):
        """
        Returns the UCT exploration value for a given state and action.

        Args:
            s: The parent node representing the state.
            a: The child node representing the action.
        """
        if (s.visits) / (1 + a.visits) <= 1:
            return 0
        return config.c * np.sqrt(math.log((s.visits) / (1 + a.visits)))

    def search(self, random_move=False):
        """
        Performs the Monte Carlo Tree Search.

        Args:
            random_move: Whether to select moves randomly during the search.

        Returns:
            The best action to take based on the search results.
        """
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
        """
        Updates the root node based on the selected move.

        Args:
            move: The move to update the root node with.

        Raises:
            ValueError: If the move is invalid.
        """
        move_found = False
        for child in self.root.children:
            if child.action == move:
                self.root = child
                move_found = True
                break
        if not move_found:
            raise ValueError(f"Invalid move: {move}")

    def extract_training_data(self):
        """
        Extracts training data from the search tree.

        Returns: Input data and predicted probabilities for training the actor network.
        """
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


