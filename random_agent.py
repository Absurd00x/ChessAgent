from random import choice

class RAgent:
    def __init__(self, board):
        self.board = board

    def make_move(self, obs=None, legal_mask=None):
        moves = list(self.board.legal_moves)
        return choice(moves)
