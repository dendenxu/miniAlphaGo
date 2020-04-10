import sys
import numpy as np
import random


class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color, depth=6):
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
        self.maxdepth = depth

    def evaluate(self, board, actions_diff, color):
        pos_color = "O" if color == "X" else "X"
        weight_list = [[90, -60, 20, 20, 20, 20, -60, 90],
                       [-60, -200, 5, 5, 5, 5, -200, 60],
                       [10, 5, 1, 1, 1, 1, 5, 10],
                       [20, 5, 1, 1, 1, 1, 5, 20],
                       [20, 5, 1, 1, 1, 1, 5, 20],
                       [10, 5, 1, 1, 1, 1, 5, 10],
                       [-60, -200, 5, 5, 5, 5, -200, 60],
                       [90, -60, 20, 20, 20, 20, -60, 90]]
        n = board._board
        weight_list = np.array(weight_list)

        if (n[0][0] != color and (n[0][1] == color or n[1][0] == color)):
            return self.minsize

        if (n[7][0] != color and (n[7][1] == color or n[6][0] == color)):
            return self.minsize

        if (n[0][7] != color and (n[0][6] == color or n[1][7] == color)):
            return self.minsize

        if (n[7][7] != color and (n[7][6] == color or n[6][7] == color)):
            return self.minsize
        if (board.count(color) + board.count(pos_color) > 20):
            if (n[0][1] == pos_color or n[1][0] == pos_color):
                weight_list[0][0] = 1000
            if (n[7][1] == pos_color or n[6][0] == pos_color):
                weight_list[7][0] = 1000
            if (n[0][6] == pos_color or n[1][7] == pos_color):
                weight_list[0][7] = 1000
            if (n[7][6] == pos_color or n[6][7] == pos_color):
                weight_list[7][7] = 1000

            if (n[0][0] != color and n[7][0] != color):
                weight_list[1:7, 0] = -1000
            if (n[0][7] != color and n[7][7] != color):
                weight_list[1:7, 7] = -1000
            if (n[0][0] != color and n[0][7] != color):
                weight_list[0, 1:7] = -1000
            if (n[7][0] != color and n[7][7] != color):
                weight_list[7, 1:7] = -1000
            if (n[0][0] != color and n[7][7] != color):
                weight_list[range(1, 7), range(1, 7)] = -1000
            if (n[7][0] != color and n[0][7] != color):
                weight_list[range(1, 7), range(6, 0, -1)] = -1000
        _board = [[1 if i == color else -1 if i == pos_color else 0 for i in l] for l in n]

        _board = np.array(_board)
        res = _board * weight_list
        # 行动力
        res = np.sum(np.sum(res)) + np.sum(_board) * 1000 + actions_diff * 500
        # 稳定子
        return res

    def move_alpha_beta(self, board, alpha, beta, color, depth):

        MAX = self.minsize
        pos_color = "O" if color == "X" else "X"
        my_actions = list(board.get_legal_actions(color))
        pos_actions = list(board.get_legal_actions(pos_color))
        cur_action = None
        action_diff = len(my_actions) - len(pos_actions)
        if (depth <= 0):
            return self.evaluate(board, action_diff, color), cur_action
        if (len(my_actions) == 0):
            if (len(pos_actions) == 0):
                return self.evaluate(board, action_diff, color), cur_action
            buffer, cur_action = self.move_alpha_beta(board, -beta, -alpha, pos_color, depth)
            return -buffer, cur_action
        for each_move in my_actions:
            flip_list = board._move(each_move, color)
            buffer, _ = self.move_alpha_beta(board, -beta, -alpha, pos_color, depth - 1)
            buffer *= -1
            board.backpropagation(each_move, flip_list, color)
            if buffer > alpha:
                if (buffer >= beta):  # 剪枝  good
                    cur_action = each_move
                    return buffer, cur_action

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
        if step < 20 and step % 2 == 0:

            return random.choice(my_actions)
        else:
            _, action = self.move_alpha_beta(board, self.minsize, self.maxsize, self.color, self.maxdepth)
        # ------------------------------------------------------------------------
        if len(my_actions) > 0:
            if action not in my_actions:
                action = random.choice(my_actions)
        else:
            action = None

        return action
