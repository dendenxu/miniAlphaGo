import numpy as np
from copy import deepcopy
from multiprocessing import Process, Manager


class AIPlayer:
    def __init__(self, color, big_val=1e10, small_val=-1e10, max_depth=3):
        self.action = None
        self.color = color
        self.opp_color = "X" if color is "O" else "O"
        self.big_val = big_val
        self.small_val = small_val
        self.depth = max_depth
        self.weight = (np.asarray([[90, -30, 10, 10, 10, 10, -30, 90],
                                   [-30, -80, 5, 5, 5, 5, -80, -30],
                                   [10, 5, 1, 1, 1, 1, 5, 10],
                                   [10, 5, 1, 1, 1, 1, 5, 10],
                                   [10, 5, 1, 1, 1, 1, 5, 10],
                                   [10, 5, 1, 1, 1, 1, 5, 10],
                                   [-30, -80, 5, 5, 5, 5, -80, -30],
                                   [90, -30, 10, 10, 10, 10, -30, 90]]))
        self.factor = abs(np.average(np.average(self.weight)))

    def get_move(self, board):
        player_name = '黑棋' if self.color == 'X' else '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        moves = list(board.get_legal_actions(self.color))
        jobs = []
        result_list = Manager().list([0 for _ in enumerate(moves)])
        for i, move in enumerate(moves):
            temp_board = deepcopy(board)
            temp_board._move(move, self.color)
            p = Process(target=self.wrapper, args=(
                temp_board, -self.big_val, -self.small_val, self.opp_color, self.depth - 1, i, result_list))
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
        return moves[np.argmax(result_list)]

    def wrapper(self, board, alpha, beta, color, depth, i, result_list):
        result = -self.alpha_beta(board, alpha, beta, color, depth)[0]
        result_list[i] = result

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

    def alpha_beta(self, board, alpha, beta, color, depth):
        action = None
        opp_color = "X" if color is "O" else "O"
        max_val = self.small_val
        moves = list(board.get_legal_actions(color))
        opp_moves = list(board.get_legal_actions(opp_color))
        if depth <= 0:
            mobility = (len(moves) - len(opp_moves)) * self.factor
            return self.evaluate(board, color, opp_color) + mobility, action
        if len(moves) is 0:
            if len(opp_moves) is 0:
                mobility = (len(moves) - len(opp_moves)) * self.factor
                return self.evaluate(board, color, opp_color) + mobility, action
            return -self.alpha_beta(board, -beta, -alpha, opp_color, depth)[0], action
        for move in moves:
            flipped = board._move(move, color)
            val = -self.alpha_beta(board, -beta, -alpha, opp_color, depth - 1)[0]
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
