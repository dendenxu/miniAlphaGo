import numpy as np
from board import Board


class AIPlayer:
    def __init__(self, color, max_val=1e10, min_val=-1e10, max_depth=3):
        self.action = None
        self.color = color
        self.max_val = max_val
        self.min_val = min_val
        self.depth = max_depth
        self.weight = np.asarray([[90, -60, 10, 10, 10, 10, -60, 90], [-60, -80, 5, 5, 5, 5, -80, -60], [10, 5, 1, 1, 1, 1, 5, 10], [10, 5, 1, 1, 1, 1, 5, 10], [10, 5, 1, 1, 1, 1, 5, 10], [10, 5, 1, 1, 1, 1, 5, 10], [-60, -80, 5, 5, 5, 5, -80, -60], [90, -60, 10, 10, 10, 10, -60, 90], ])

    def get_move(self, board):
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        self.action = None
        self.alpha_beta(board, self.min_val, self.max_val, self.color, self.depth)
        print(self.action)
        return self.action

    def oppo_color(self, color):
        return "X" if color is "O" else "O"

    def sign_color(self, color):
        return 1 if color is self.color else - 1

    def evaluate(self, board, color):
        _board = np.sum(np.asarray([[self.sign_color(i) for i in j] for j in board._board]) * self.weight)
        return self.sign_color(color) * _board

    def alpha_beta(self, board, alpha, beta, color, depth):
        max_val = self.min_val
        moves = list(board.get_legal_actions(color))
        pos_moves = list(board.get_legal_actions(self.oppo_color(color)))
        if depth <= 0:
            return self.evaluate(board, color)
        if len(moves) is 0:
            if len(pos_moves) is 0:
                return self.evaluate(board, color)
            return -self.alpha_beta(board, -beta, -alpha, self.oppo_color(color), depth)
        for move in moves:
            flipped = board._move(move, color)
            val = -self.alpha_beta(board, -beta, -alpha, self.oppo_color(color), depth - 1)
            print(depth, end=' ')
            print(self.action)
            print(moves)
            board.backpropagation(move, flipped, color)
            if val > max_val:
                max_val = val
                self.action = move
            if max_val > alpha:
                if max_val >= beta:
                    self.action = move
                    return max_val
                alpha = max_val
        return max_val
