import sys
import numpy as np


class SimpleAIPlayer:
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

    def evaluate(self, board, actions_diff):
        weight_list = np.asarray([[90, -60, 10, 10, 10, 10, -60, 90], [-60, -80, 5, 5, 5, 5, -80, -60], [10, 5, 1, 1, 1, 1, 5, 10], [10, 5, 1, 1, 1, 1, 5, 10], [10, 5, 1, 1, 1, 1, 5, 10], [10, 5, 1, 1, 1, 1, 5, 10], [-60, -80, 5, 5, 5, 5, -80, -60], [90, -60, 10, 10, 10, 10, -60, 90], ])
        n = board._board
        _board = [[1 if i == self.color else -1 if i == self.poscolor else 0 for i in l] for l in n]
        _board = np.asarray(_board)
        # print("_board is\n {}".format(_board))
        res = _board*weight_list
        # print("res is {} and actions_diff is {}".format(res, actions_diff))
        res = np.sum(np.sum(res))+actions_diff*np.average(np.average(weight_list))
        # print("res is {} and actions_diff is {}".format(res, actions_diff))
        return res

    def move_alpha_beta(self, board, alpha, beta, color, depth):
        MAX = self.minsize
        pos_color = "O" if color == "X" else "X"
        my_actions = list(board.get_legal_actions(color))
        pos_actions = list(board.get_legal_actions(pos_color))
        cur_action = None
        action_diff = self.sign[color]*(len(my_actions)-len(pos_actions))
        if(depth <= 0):
            return self.sign[color]*self.evaluate(board, action_diff), cur_action
        if(len(my_actions) == 0):
            if(len(pos_actions) == 0):
                return self.sign[color]*self.evaluate(board, action_diff), cur_action
            buffer, cur_action = self.move_alpha_beta(board, -beta, -alpha, pos_color, depth)
            return -buffer, cur_action
        for each_move in my_actions:
            flip_list = board._move(each_move, color)
            buffer, _ = self.move_alpha_beta(board, -beta, -alpha, pos_color, depth-1)
            buffer *= -1
            board.backpropagation(each_move, flip_list, color)
            if buffer > alpha:
                if(buffer >= beta):  # 剪枝
                    cur_action = each_move
                    return buffer, cur_action
                alpha = max(buffer, alpha)  # 更新alpha
            if(buffer > MAX):
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

        _, action = self.move_alpha_beta(board, self.minsize, self.maxsize, self.color, 3)
        # ------------------------------------------------------------------------
        print(action)
        return action
