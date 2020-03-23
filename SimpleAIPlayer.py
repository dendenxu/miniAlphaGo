import numpy as np


class AIPlayer:
    '''An AIPlayer that gives moves according to current board status'''

    def __init__(self, color, big_val=1e10, small_val=-1e10, max_depth=6, max_width=12):
        self.color = color  # Defining the color of the current player, use 'X' or 'O'
        self.oppo_color = "X" if color is "O" else "O"  # Precomputing opponent's color for performance
        self.big_val = big_val  # A value big enough for beta initialization
        self.small_val = small_val  # A value small enough for alpha initialization
        self.depth = max_depth  # Max search depth of game tree
        self.max_width = max_width  # Max search width of game tree, not used in practice
        self.weight = np.asarray([[150, -80, 10, 10, 10, 10, -80, 150], # weight matrix of board position
                                  [-80, -90, 5, 5, 5, 5, -90, -80],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [-80, -90, 5, 5, 5, 5, -90, -80],
                                  [150, -80, 10, 10, 10, 10, -80, 150]])
        self.factor = 50  # How much mobility and stability affects the evaluation of current board

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------

        result = self.alpha_beta(board, self.small_val, self.big_val, self.color, self.depth, board.count("X") + board.count("O"))

        # print(result)
        # ------------------------------------------------------------------------
        return result[1]

    def evaluate(self, board, color, oppo_color):
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

        _board *= self.weight
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

