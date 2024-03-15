class actor(): 
    def __init__(self, ann):
        self.ann = ann

    def makeMove(self, state):
        move_probabilities = self.ann.compute_move_probabilities(state)
        return max(move_probabilities)