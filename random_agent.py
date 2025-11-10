from random import choice

class RAgent:
    def __init__(self, board):
        self.board = board

    def __call__(self, obs=None):
        moves = list(self.board.legal_moves)
        return choice(moves)
