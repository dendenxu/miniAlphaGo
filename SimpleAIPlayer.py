import numpy as np
import time
from board import Board


class AIPlayer:
    def __init__(self, color, big_val=1e10, small_val=-1e10, max_depth=3):
        self.action = None
        self.color = color
        self.oppo_color = "X" if color is "O" else "O"
        self.big_val = big_val
        self.small_val = small_val
        self.depth = max_depth
        self.weight = np.ones((8, 8), int)

    def get_move(self, board):
        player_name = '黑棋' if self.color == 'X' else '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        start_time = time.perf_counter()
        result = self.alpha_beta(board, self.small_val, self.big_val, self.color, self.depth)
        end_time = time.perf_counter()
        print("Time used in calculating result is: {}".format(end_time - middle_time))
        # print(result)
        return result[1]

    def evaluate(self, board, color):
        _board = np.asarray([[1 if (piece is self.color) else (-1 if piece is self.oppo_color else 0) for piece in line] for line in board._board])
        _board *= self.weight
        _board = np.sum(_board)
        return (1 if color is self.color else - 1) * _board

    def alpha_beta(self, board, alpha, beta, color, depth):
        action = None
        oppo_color = "X" if color is "O" else "O"
        max_val = self.small_val
        moves = list(board.get_legal_actions(color))
        oppo_moves = list(board.get_legal_actions(oppo_color))
        if depth <= 0:
            return self.evaluate(board, color), action
        if len(moves) is 0:
            if len(oppo_moves) is 0:
                return self.evaluate(board, color), action
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
