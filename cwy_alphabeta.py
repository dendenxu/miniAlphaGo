import math
import random
from func_timeout import func_timeout, FunctionTimedOut
import datetime

class AlphaBeta_AI:
    """
    alpha-beta pruning
    """
    def __init__(self, color, max_depth = 1):
        self.color = color
        self.max_depth = max_depth
        self.op_color = 'O' if self.color == 'X' else 'X'
        
    def terminal_test(self, board):
        """
        判断游戏是否结束
        :return: True/False 游戏结束/游戏没有结束
        """
        b_list = list(board.get_legal_actions('X'))
        w_list = list(board.get_legal_actions('O'))

        is_over = len(b_list) == 0 and len(w_list) == 0
        return is_over
    
    def utility(self, board):
        """
        游戏结束时的数值
        :return: 0-黑棋赢，1-白旗赢，2-表示平局，黑棋个数和白旗个数相等
        """
        winner, diff = board.get_winner()
        if winner == 0 and self.color == 'X':
            return diff
        elif winner == 1 and self.color == 'O':
            return diff
        elif winner == 2:
            return 0
        else:
            return -diff
        
    def maxvalue(self, board, depth, alpha, beta):
        if self.terminal_test(board):
            return self.utility(board)
        if depth == 0:
            return self.utility(board)
        v = -math.inf
        for a in list(board.get_legal_actions(self.color)):
            flipped_pos = board._move(a, self.color)
            v = max(v, self.minvalue(board, depth - 1, alpha, beta))
            board.backpropagation(a, flipped_pos, self.color)
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
        
    
    def minvalue(self, board, depth, alpha, beta):
        if self.terminal_test(board):
            return self.utility(board)
        if depth == 0:
            return self.utility(board)
        v = math.inf
        for a in list(board.get_legal_actions(self.op_color)):
            flipped_pos = board._move(a, self.op_color)
            v = min(v, self.maxvalue(board, depth - 1, alpha, beta))
            board.backpropagation(a, flipped_pos, self.op_color)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def alphabetasearch(self, board):
        best_action = []
        best_val = -math.inf
        for a in list(board.get_legal_actions(self.color)):
            #start_time = datetime.datetime.now()
            flipped_pos = board._move(a, self.color)
            val = self.minvalue(board, self.max_depth - 1, alpha = -math.inf, beta = math.inf)
            board.backpropagation(a, flipped_pos, self.color)
            if val > best_val:
                best_val = val
                best_action = [a]
            elif val == best_val:
                best_action.append(a)
        if(len(best_action) > 1):
            return random.choice(best_action)
        else:
            return best_action[0]

class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color
        

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

        action = None
        if len(list(board.get_legal_actions(self.color))) != 0:
            ab = AlphaBeta_AI(self.color, 5);
            action = ab.alphabetasearch(board)
        
        return action