import numpy as np


class AIPlayer:

    def __init__(self, color, big_val=1e10, small_val=-1e10, max_depth=7, max_width=12):
        self.color = color
        self.oppo_color = "X" if color is "O" else "O"
        self.big_val = big_val
        self.small_val = small_val
        self.depth = max_depth
        self.max_width = max_width
        self.goods = ["A1", "A8", "H1", "H8"]
        self.bads = ["A2", "B1", "B2", "A7", "B7", "B8", "G1", "H2", "G2", "G7", "G8", "H7"]
        # self.weight = np.asarray([[500, -60, 20, 20, 20, 20, -60, 500],
        #                           [-60, -500, 5, 5, 5, 5, -500, -60],
        #                           [10, 5, 1, 1, 1, 1, 5, 10],
        #                           [20, 5, 1, 1, 1, 1, 5, 20],
        #                           [20, 5, 1, 1, 1, 1, 5, 20],
        #                           [10, 5, 1, 1, 1, 1, 5, 10],
        #                           [-60, -500, 5, 5, 5, 5, -500, -60],
        #                           [500, -60, 20, 20, 20, 20, -60, 500]])
        self.weight = np.asarray([[150, -80, 10, 10, 10, 10, -80, 150],
                                  [-80, -90, 5, 5, 5, 5, -90, -80],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 1, 1, 5, 10],
                                  [-80, -90, 5, 5, 5, 5, -90, -80],
                                  [150, -80, 10, 10, 10, 10, -80, 150]])

        self.factor = 50
        self.history = np.tile(np.arange(64), 128).reshape((2, 64, 64))

    def get_move(self, board):
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        moves = list(board.get_legal_actions(self.color))
        global_depth = board.count("X") + board.count("O")
        for good in self.goods:
            if good in moves:
                return good
        _, result = self.alpha_beta(board, self.small_val, self.big_val, self.color, self.depth,
                                    global_depth)
        if len(moves) != 0 and result not in moves:
            return moves[0]
        return result

    def evaluate(self, board, color, oppo_color):
        _board = np.asarray([[1 if (piece is color) else (-1 if piece is oppo_color else 0)
                              for piece in line] for line in board._board])

        n = _board
        if n[0][0] is not 1 and (n[0][1] or n[1][0] or n[1][1]):
            return self.small_val
        if n[7][0] is not 1 and (n[7][1] or n[6][0] or n[6][1]):
            return self.small_val
        if n[0][7] is not 1 and (n[0][6] or n[1][7] or n[1][6]):
            return self.small_val
        if n[7][7] is not 1 and (n[7][6] or n[6][7] or n[6][6]):
            return self.small_val

        _board *= self.weight
        _board = np.sum(_board)

        return _board

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
            if global_depth < 16:
                return mobility, action
            return self.evaluate(board, color, oppo_color) + mobility, action
        if depth == self.depth and global_depth <= 52:
            moves = self.remove_bad(moves)
        moves = self.history_sort(board, moves, color, global_depth)
        for move in moves:
            flipped = board._move(move, color)
            val = -(self.alpha_beta(board, -beta, -alpha, oppo_color, depth - 1, step)[0])
            board.backpropagation(move, flipped, color)
            if val > max_val:
                max_val = val
                action = move
                if max_val > alpha:
                    if max_val >= beta:
                        self.reward_move(board, action, color, global_depth, True)
                        return max_val, action
                    alpha = max_val
        if action is not None:
            self.reward_move(board, action, color, global_depth, False)
        return max_val, action

    def remove_bad(self, moves):
        temp_moves = moves.copy()
        for bad in self.bads:
            try:
                temp_moves.remove(bad)
            except ValueError:
                pass
        if len(temp_moves) != 0:
            return temp_moves
        else:
            return moves

    def history_sort(self, board, moves, color, depth):
        if depth >= 64:
            return moves
        poss = list(map(lambda x: x[0] * 8 + x[1], [board.board_num(move) for move in moves]))
        values = self.history[int(color == self.color), depth][poss]
        idx = np.argsort(values)
        return np.asarray(moves)[idx]

    def reward_move(self, board, move, color, depth, best):
        if depth >= 64:
            return
        x, y = board.board_num(move)
        pos = x * 8 + y
        color = int(color == self.color)
        val = self.history[color, depth, pos]
        other_pos = np.argwhere(self.history[color, depth, :] == val - (val if best else (1 if val else 0)))
        self.history[color, depth, other_pos], self.history[color, depth, pos] = \
            self.history[color, depth, pos], self.history[color, depth, other_pos]
