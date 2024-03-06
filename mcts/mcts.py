import math
import random
import numpy as np
from anytree import Node, RenderTree
from game_logic.nim_state_manager import NIM_STATE_MANAGER
    
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

class MonteCarloTreeSearch:
    def __init__(self, root_state, actor_network, state_manager):
        self.root = MCTSNode(root_state)
        self.actor_network = actor_network
        self.state_manager = state_manager

    def select_node(self) -> MCTSNode:
        node = self.root
        while node.children:
            node.visits += 1
            node = self.tree_policy(node)
        return node
        
    def expand_node(self, node):
        node.visits += 1
        self.state_manager.setState(node.state)
        legal_moves = self.state_manager.getLegalMoves(node.state)
        for move in legal_moves:
            new_state = self.state_manager.simulateMove(move, node.state)
            MCTSNode(new_state, parent=node, action=move)


    def rollout(self, node):
        # Simulate a game from the current state
        current_state = node.state
        while not self.state_manager.isGameOver(current_state):
            legal_moves = self.state_manager.getLegalMoves(current_state)
            # TODO: Replace with actor network
            move = random.choice(legal_moves)
            current_state = self.state_manager.simulateMove(move, current_state)
        self.backpropagate(node, self.state_manager.getReward(node.state))

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
        return sorted(self.root.children, key=lambda c: c.visits)[-1].action
        
    def tree_policy(self, node):
        # Return the best child node according to the UCT policy
        max_score = -np.inf
        best_child = []
        for child in node.children:
            score = MonteCarloTreeSearch.Q(child, node.state[1]) + MonteCarloTreeSearch.u(node, child)
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
        return 0.2*np.sqrt(math.log((s.visits) /(1 + a.visits)))

    def search(self):
        for _ in range(100000):
            node = self.select_node()
            self.expand_node(node)
            self.rollout(node)
        return self.best_action()

