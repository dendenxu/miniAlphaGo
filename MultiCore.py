import numpy as np
from copy import deepcopy
from multiprocessing import Process, Manager


class AIPlayer:
    def __init__(self, color, big_val=1e10, small_val=-1e10, max_depth=6, max_width=12):
        self.action = None
        self.color = color
        self.oppo_color = "X" if color is "O" else "O"
        self.big_val = big_val
        self.small_val = small_val
        self.depth = max_depth
        self.max_width = max_width
        self.weight = np.asarray([[90, -60, 10, 10, 10, 10, -60, 90],
                                  [-60, -80, 5, 5, 5, 5, -80, 60],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [-60, -80, 5, 5, 5, 5, -80, 60],
                                  [90, -60, 10, 10, 10, 10, -60, 90]])
        self.factor = abs(np.average(self.weight)) * 50

    def get_move(self, board):
        moves = list(board.get_legal_actions(self.color))
        jobs = []
        result_list = Manager().list(range(len(moves)))
        step = board.count("X") + board.count("O")
        for i, move in enumerate(moves):
            temp_board = deepcopy(board)
            temp_board._move(move, self.color)
            p = Process(target=self.wrapper, args=(
                temp_board, self.small_val, self.big_val, self.oppo_color, self.depth - 1, step, i, result_list))
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
        idx = np.argmax(result_list)
        print((result_list[idx], moves[idx]))
        return moves[idx]

    def wrapper(self, board, alpha, beta, color, depth, step, i, result_list):
        result_list[i] = -self.alpha_beta(board, alpha, beta, color, depth, step)[0]

    def evaluate(self, board, color, oppo_color):
        weight = self.weight
        _board = np.asarray([[1 if (piece is color) else (-1 if piece is oppo_color else 0)
                              for piece in line] for line in board._board])
        sep_board = np.stack(((_board == 1).astype(int), np.negative((_board == -1).astype(int))))
        stability = 0
        for i in range(2):
            if sep_board[i, 0, 0]:
                stability += np.sum(sep_board[i, 0, 0:-1]) + np.sum(sep_board[i, 1::, 0])
                if sep_board[i, 1, 1]:
                    stability += np.sum(sep_board[i, 1, 1:-2]) + np.sum(sep_board[i, 2:-1, 1])
            if sep_board[i, 0, -1]:
                stability += np.sum(sep_board[i, 0:-1, -1]) + np.sum(sep_board[i, 0, 0:-1])
                if sep_board[i, 1, -2]:
                    stability += np.sum(sep_board[i, 1:-2, -2]) + np.sum(sep_board[i, 1, 1:-2])
            if sep_board[i, -1, -1]:
                stability += np.sum(sep_board[i, -1, 1::]) + np.sum(sep_board[i, 0:-1, -1])
                if sep_board[i, -2, -2]:
                    stability += np.sum(sep_board[i, -2, 2:-1]) + np.sum(sep_board[i, 1:-2, -2])
            if sep_board[i, -1, 0]:
                stability += np.sum(sep_board[i, 1::, 0]) + np.sum(sep_board[i, -1, 1::])
                if sep_board[i, -2, 1]:
                    stability += np.sum(sep_board[i, 2:-1, 1]) + np.sum(sep_board[i, -2, 2:-1])

        _board *= weight
        _board = np.sum(_board)
        _board += stability * self.factor
        return _board if np.sum(sep_board[0, :, :]) else self.small_val

    def alpha_beta(self, board, alpha, beta, color, depth, step):
        action = None
        oppo_color = "X" if color is "O" else "O"
        max_val = self.small_val
        moves = list(board.get_legal_actions(color))
        global_depth = step + self.depth - depth
        oppo_moves = list(board.get_legal_actions(oppo_color))

        if len(moves) is 0:
            if len(oppo_moves) is 0:
                mobility = (len(moves) - len(oppo_moves)) * self.factor
                return self.evaluate(board, color, oppo_color) + mobility, action
            return -self.alpha_beta(board, -beta, -alpha, oppo_color, depth, step)[0], action
        if depth <= 0:
            mobility = (len(moves) - len(oppo_moves)) * self.factor
            if global_depth < 22:
                return mobility, action
            return self.evaluate(board, color, oppo_color) + mobility, action

        # moves = moves[::min(len(moves), self.max_width)]
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
                    return max_val, action
                alpha = max_val
        return max_val, action
