import numpy as np
from board import Board


class AIPlayer:
    def __init__(self, color, max_val=1e10, min_val=-1e10, max_depth=3):
        self.action = None
        self.color = color
        self.oppo_color = "X" if color is "O" else "O"
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
        return self.alpha_beta(board, self.min_val, self.max_val, self.color, self.depth)[1]

    def sign_color(self, color):
        return 1 if color is self.color else - 1

    def value_color(self, color):
        return 1 if color is self.color else (-1 if color is self.oppo_color(self.color) else 0)

    def evaluate(self, board, color):
        _board = np.asarray([[1 if (i is self.color) else (-1 if i is self.oppo_color else 0) for i in j] for j in board._board])
        _board *= self.weight
        _board = np.sum(_board)
        # print("_board is\n {}".format(_board))
        return (1 if color is self.color else - 1) * _board

    def alpha_beta(self, board, alpha, beta, color, depth):
        oppo_color = "X" if color is "O" else "O"
        # print(color)
        action = None
        max_val = self.min_val
        moves = list(board.get_legal_actions(color))
        oppo_moves = list(board.get_legal_actions(oppo_color))
        movability = len(moves) - len(oppo_moves)
        movability *= 1 if color is self.color else - 1
        movability *= np.average(np.average(self.weight))
        if depth <= 0:
            return self.evaluate(board, color) + movability, action
        if len(moves) is 0:
            if len(oppo_moves) is 0:
                return self.evaluate(board, color) + movability, action
            return -self.alpha_beta(board, -beta, -alpha, oppo_color, depth)[0], action
        for move in moves:
            flipped = board._move(move, color)
            val = -self.alpha_beta(board, -beta, -alpha, oppo_color, depth - 1)[0]
            board.backpropagation(move, flipped, color)
            if val > max_val:
                max_val = val
                action = move
            if max_val > alpha:
                if max_val >= beta:
                    action = move
                    return max_val, action
                alpha = max_val
        return max_val, action
