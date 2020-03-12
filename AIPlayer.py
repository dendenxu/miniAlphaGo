import numpy as np
from board import Board
from copy import deepcopy
from threading import Thread


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class AIPlayer:
    def __init__(self, color, big_val=1e10, small_val=-1e10, max_depth=3):
        self.action = None
        self.color = color
        self.oppo_color = "X" if color is "O" else "O"
        self.big_val = big_val
        self.small_val = small_val
        self.depth = max_depth
        self.weight = np.asarray([[90, -60, 10, 10, 10, 10, -60, 90],
                                  [-60, -80, 5, 5, 5, 5, -80, -60],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [-60, -80, 5, 5, 5, 5, -80, -60],
                                  [90, -60, 10, 10, 10, 10, -60, 90], ])

    def get_move(self, board):
        player_name = '黑棋' if self.color == 'X' else '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        moves = list(board.get_legal_actions(self.color))
        pool = []
        for _, move in enumerate(moves):
            temp_board = deepcopy(board)
            temp_board._move(move, self.color)
            pool.append(ThreadWithReturnValue(target=self.alpha_beta, args=(temp_board, -self.big_val, -self.small_val, self.oppo_color, self.depth)))
            pool[-1].start()
        action = None
        result = np.zeros((len(moves)))
        for i, _ in enumerate(moves):
            result[i] = -pool[i].join()[0]
        action = moves[np.argmax(result)]
        return action

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
        movability = len(moves) - len(oppo_moves) * (1 if color is self.color else - 1)
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
