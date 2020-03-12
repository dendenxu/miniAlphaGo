import numpy as np
from random import shuffle


class AIPlayer:
    def __init__(self, color, big_val=1e10, small_val=-1e10, max_depth=3, max_width=10):
        self.action = None
        self.color = color
        self.oppo_color = "X" if color is "O" else "O"
        self.big_val = big_val
        self.small_val = small_val
        self.depth = max_depth
        self.max_width = max_width
        self.weight = (np.asarray([[90, -30, 10, 10, 10, 10, -30, 90],
                                   [-30, -80, 5, 5, 5, 5, -80, -30],
                                   [10, 5, 1, 1, 1, 1, 5, 10],
                                   [10, 5, 1, 1, 1, 1, 5, 10],
                                   [10, 5, 1, 1, 1, 1, 5, 10],
                                   [10, 5, 1, 1, 1, 1, 5, 10],
                                   [-30, -80, 5, 5, 5, 5, -80, -30],
                                   [90, -30, 10, 10, 10, 10, -30, 90]]))
        self.factor = abs(np.average(np.average(self.weight)))
        self.history = np.tile(np.arange(64), 128).reshape((2, 64, 64))

    def get_move(self, board):
        player_name = '黑棋' if self.color == 'X' else '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        return self.alpha_beta(board, self.small_val, self.big_val,
                               self.color, self.depth, board.count("X") + board.count("O"))[1]

    def evaluate(self, board, color, oppo_color):
        weight = self.weight
        _board = np.asarray([[1 if (piece is color) else (-1 if piece is oppo_color else 0)
                              for piece in line] for line in board._board])
        sep_board = np.stack(((_board > 1).astype(int), np.negative((_board < -1).astype(int))))
        stability = 0
        for i in range(2):
            if sep_board[0, 0, i]:
                stability += np.sum(sep_board[0, 0::-1, i])
            if sep_board[0, -1, i]:
                stability += np.sum(sep_board[0::-1, -1, i])
            if sep_board[-1, 0, i]:
                stability += np.sum(sep_board[-1, 1::, i])
            if sep_board[-1, -1, i]:
                stability += np.sum(sep_board[1::, 0, i])
        _board *= weight
        _board = np.sum(_board)
        _board += stability * self.factor
        return _board

    def alpha_beta(self, board, alpha, beta, color, depth, step):
        action = None
        oppo_color = "X" if color is "O" else "O"
        max_val = self.small_val
        moves = list(board.get_legal_actions(color))
        oppo_moves = list(board.get_legal_actions(oppo_color))
        if depth <= 0:
            mobility = (len(moves) - len(oppo_moves)) * self.factor
            return self.evaluate(board, color, oppo_color) + mobility, action
        if len(moves) is 0:
            if len(oppo_moves) is 0:
                mobility = (len(moves) - len(oppo_moves)) * self.factor
                return self.evaluate(board, color, oppo_color) + mobility, action
            return -self.alpha_beta(board, -beta, -alpha, oppo_color, depth, step)[0], action
        global_depth = step + self.depth - depth
        moves = self.history_sort(board, moves, color, global_depth)
        moves = moves[0:min(len(moves), self.max_width)]
        for move in moves:
            flipped = board._move(move, color)
            val = -self.alpha_beta(board, -beta, -alpha, oppo_color, depth - 1, step)[0]
            board.backpropagation(move, flipped, color)
            if val > max_val:
                max_val = val
                action = move
            if max_val > alpha:
                if max_val >= beta:
                    action = move
                    self.reward_move(board, action, color, global_depth, True)
                    return max_val, action
                alpha = max_val
        self.reward_move(board, action, color, global_depth, False)
        return max_val, action

    def history_sort(self, board, moves, color, depth):
        poss = list(map(lambda x: x[0] * 8 + x[1], [board.board_num(move) for move in moves]))
        values = self.history[int(color == self.color), depth][poss]
        idx = np.argsort(values)
        return np.asarray(moves)[idx]

    def reward_move(self, board, move, color, depth, best):
        x, y = board.board_num(move)
        pos = x * 8 + y
        color = int(color == self.color)
        val = self.history[color, depth, pos]
        other_pos = np.argwhere(self.history[color, depth, :] ==
                                val - (val if best else (1 if val else 0)))
        self.history[color, depth, other_pos], \
        self.history[color, depth, pos] = \
            self.history[color, depth, pos], \
            self.history[color, depth, other_pos]
