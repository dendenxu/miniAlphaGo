import sys
import numpy as np


class AIPlayer_history:
    """
    AI 玩家
    """

    def __init__(self, color):
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
        self.history = [[list(range(65)) for i in range(65)] for i in range(2)]

    def number2cor(self, num):
        return num//8, num % 8

    def cor2number(self, action):
        x, y = action
        return x*8+y

    def moveTop(self, color, step, num_cor):
        action_index = self.history[self.index[color]][step].index(num_cor)
        self.history[self.index[color]][step][action_index] = self.history[self.index[color]][step][0]
        self.history[self.index[color]][step][0] = num_cor

    def movenext(self, color, step, num_cor):
        action_index = self.history[self.index[color]][step].index(num_cor)
        if(not action_index == 0):
            self.history[self.index[color]][step][action_index] = self.history[self.index[color]][step][action_index-1]
            self.history[self.index[color]][step][action_index-1] = num_cor

    def evaluate(self, board, actions_diff, color):
        pos_color = "O" if color == "X" else "X"
        weight_list = [[90, -60, 10, 10, 10, 10, -60, 90], [-60, -80, 5, 5, 5, 5, -80, 60], [10, 5, 1, 1, 1, 1, 5, 10],
                       [10, 5, 1, 1, 1, 1, 5, 10],
                       [10, 5, 1, 1, 1, 1, 5, 10], [10, 5, 1, 1, 1, 1, 5, 10], [-60, -80, 5, 5, 5, 5, -80, 60],
                       [90, -60, 10, 10, 10, 10, -60, 90]]
        n = board._board
        _board = [[1 if i == color else -1 if i == pos_color else 0 for i in l] for l in n]
        flag = [True, True, True, True]
        for i in range(1, 8):
            if (_board[0][i] == _board[0][i - 1] and flag[0]):
                weight_list[0][i] = weight_list[0][i - 1]
            else:
                flag[0] = False
            if (_board[7][i] == _board[7][i - 1] and flag[1]):
                weight_list[7][i] = weight_list[7][i - 1]
            else:
                flag[1] = False
            if (_board[i][0] == _board[i - 1][0] and flag[2]):
                weight_list[i][0] = weight_list[i - 1][0]
            else:
                flag[2] = False
            if (_board[i][7] == _board[i - 1][7] and flag[3]):
                weight_list[i][7] = weight_list[i - 1][7]
            else:
                flag[3] = False
        flag = [True, True, True, True]
        for i in range(6, -1, -1):
            if (_board[0][i] == _board[0][i + 1] and flag[0]):
                weight_list[0][i] = weight_list[0][i + 1]
            else:
                flag[0] = False
            if (_board[7][i] == _board[7][i + 1] and flag[1]):
                weight_list[7][i] = weight_list[7][i + 1]
            else:
                flag[1] = False
            if (_board[i][0] == _board[i + 1][0] and flag[2]):
                weight_list[i][0] = weight_list[i + 1][0]
            else:
                flag[2] = False
            if (_board[i][7] == _board[i + 1][7] and flag[3]):
                weight_list[i][7] = weight_list[i + 1][7]
            else:
                flag[3] = False

        weight_list = np.array(weight_list)
        _board = np.array(_board)
        res = _board * weight_list
        # 行动力
        res = np.sum(np.sum(res)) + actions_diff * 50
        # 稳定子
        return res

    def move_alpha_beta(self, board, alpha, beta, color, depth, step):

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
            buffer, cur_action = self.move_alpha_beta(board, -beta, -alpha, pos_color, depth, step)
            return -buffer, cur_action
        for history_action in self.history[self.index[color]][step]:
            cor = self.number2cor(history_action)
            each_move = board.num_board(cor)
            if(each_move in my_actions):
                flip_list = board._move(each_move, color)
                buffer, _ = self.move_alpha_beta(board, -beta, -alpha, pos_color, depth - 1, step+1)
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
        step = board.count('X')+board.count('O')
        _, action = self.move_alpha_beta(board, self.minsize, self.maxsize, self.color, 6, step)
        # ------------------------------------------------------------------------
        return action
