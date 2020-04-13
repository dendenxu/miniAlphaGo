import sys
import numpy as np
import random


class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color, depth=7):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color
        self.poscolor = "O" if color == "X" else "X"
        self.minsize = -999999999
        self.maxsize = 999999999
        self.sign = {}
        self.sign[self.color] = 1
        self.sign[self.poscolor] = -1
        self.index = {}
        self.index[self.poscolor] = 0
        self.index[self.color] = 1
        self.history = [[list(range(70)) for i in range(70)] for i in range(2)]
        self.maxdepth = depth

    def number2cor(self, num):
        return num // 8, num % 8

    def cor2number(self, action):
        x, y = action
        return x * 8 + y

    def moveTop(self, color, step, num_cor):
        action_index = self.history[self.index[color]][step].index(num_cor)
        self.history[self.index[color]][step][action_index] = self.history[self.index[color]][step][0]
        self.history[self.index[color]][step][0] = num_cor

    def movenext(self, color, step, num_cor):
        action_index = self.history[self.index[color]][step].index(num_cor)
        if (not action_index == 0):
            self.history[self.index[color]][step][action_index] = self.history[self.index[color]][step][
                action_index - 1]
            self.history[self.index[color]][step][action_index - 1] = num_cor

    def evaluate(self, board, len_action_my, len_action_pos, color):
        res = 0
        pos_color = "O" if color == "X" else "X"
        weight_list = [[0, -3, 11, 8, 8, 1, -3, 0],
                       [-3, -7, -4, 1, 1, -4, -7, -3],
                       [11, -4, 2, 2, 2, 2, -4, 11],
                       [8, 1, 2, -3, -3, 2, 1, 8],
                       [8, 1, 2, -3, -3, 2, 1, 8],
                       [11, -4, 2, 2, 2, 2, -4, 11],
                       [-3, -7, -4, 1, 1, -4, -7, -3],
                       [0, -3, 11, 8, 8, 1, -3, 0]]
        n = board._board
        weight_list = np.array(weight_list)
        unstability = 20
        if (n[0][0] != color):
            if (n[0][1] == color):
                res -= unstability
            if (n[1][0] == color):
                res -= unstability
            if (n[1][1] == color):
                res -= unstability
        else:
            res += 100
        if (n[0][0] != pos_color):
            if (n[0][1] == pos_color):
                res += unstability
            if (n[1][0] == pos_color):
                res += unstability
            if (n[1][1] == pos_color):
                res += unstability
        else:
            res -= 100
        if (n[7][0] != color):
            if (n[7][1] == color):
                res -= unstability
            if (n[6][0] == color):
                res -= unstability
            if (n[6][1] == color):
                res -= unstability
        else:
            res += 100
        if (n[7][0] != pos_color):
            if (n[7][1] == pos_color):
                res += unstability
            if (n[6][0] == pos_color):
                res += unstability
            if (n[6][1] == pos_color):
                res += unstability
        else:
            res -= 100
        if (n[0][7] != color):
            if (n[0][6] == color):
                res -= unstability
            if (n[1][7] == color):
                res -= unstability
            if (n[1][6] == color):
                res -= unstability
        else:
            res += 100
        if (n[0][7] != pos_color):
            if (n[0][6] == pos_color):
                res += unstability
            if (n[1][7] == pos_color):
                res += unstability
            if (n[1][6] == pos_color):
                res += unstability
        else:
            res -= 100
        if (n[7][7] != color):
            if (n[7][6] == color):
                res -= unstability
            if (n[6][7] == color):
                res -= unstability
            if (n[6][6] == color):
                res -= unstability
        else:
            res += 100
        if (n[7][7] != pos_color):
            if (n[7][6] == pos_color):
                res += unstability
            if (n[6][7] == pos_color):
                res += unstability
            if (n[6][6] == pos_color):
                res += unstability
        else:
            res -= 100
        _board = [[1 if i == color else -1 if i == pos_color else 0 for i in l] for l in n]
        _board = np.array(_board)
        res += np.sum(np.sum(_board * weight_list))

        # 行动力
        res += (len_action_my - len_action_pos) / res * 5 + np.sum(np.sum(_board)) / res * 5

        # 稳定子
        return res

    def move_alpha_beta(self, board, alpha, beta, color, depth, step):

        MAX = self.minsize
        pos_color = "O" if color == "X" else "X"
        my_actions = list(board.get_legal_actions(color))
        pos_actions = list(board.get_legal_actions(pos_color))
        cur_action = None
        if (depth <= 0):
            return self.evaluate(board, len(my_actions), len(pos_actions), color), cur_action
        if (len(my_actions) == 0):
            if (len(pos_actions) == 0):
                return self.evaluate(board, len(my_actions), len(pos_actions), color), cur_action
            buffer, cur_action = self.move_alpha_beta(board, -beta, -alpha, pos_color, depth, step)
            return -buffer, cur_action
        for history_action in self.history[self.index[color]][step]:
            cor = self.number2cor(history_action)
            each_move = board.num_board(cor)
            if (each_move in my_actions):
                flip_list = board._move(each_move, color)
                buffer, _ = self.move_alpha_beta(board, -beta, -alpha, pos_color, depth - 1, step + 1)
                buffer *= -1
                board.backpropagation(each_move, flip_list, color)
                cor = board.board_num(each_move)
                num_cor = self.cor2number(cor)
                if buffer > alpha:
                    if (buffer >= beta):  # 剪枝  good
                        cur_action = each_move
                        self.moveTop(color, step, num_cor)
                        return buffer, cur_action
                    self.movenext(color, step, num_cor)
                    alpha = max(buffer, alpha)  # 更新alpha
                if (buffer > MAX):
                    MAX = buffer
                    cur_action = each_move
        return MAX, cur_action

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
        step = board.count('X') + board.count('O')
        my_actions = list(board.get_legal_actions(self.color))
        _, action = self.move_alpha_beta(board, self.minsize, self.maxsize, self.color, self.maxdepth, step)
        # _, action = self.move_alpha_beta(board, self.minsize, self.maxsize, self.color, self.maxdepth,step)
        if (len(my_actions) > 0):
            if (action not in my_actions):
                action = random.choice(my_actions)
        else:
            action = None

        # ------------------------------------------------------------------------

        return action
