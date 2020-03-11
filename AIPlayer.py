import numpy as np
from board import Board


class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color, max_val=1e10, min_val=-1e10, max_depth=5):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color  # Color of the AIPlayer
        self.max_val = max_val  # A big enough value in the game to initialize "beta" or other decending value
        self.min_val = min_val  # A small enough value in the game to initialize "alpha" or other growing value
        self.depth = max_depth  # Maximum calculation depth that determine how well can an agent perform and how fast can it do it
        self.weight = np.ones((8, 8), dtype=float)
        self.weight[1:-1, 1:-1] = np.full((6, 6), 0.85)
        self.weight[2:-2, 2:-2] = np.full((4, 4), 0.75)
        self.weight[3:-3, 3:-3] = np.full((2, 2), 0.65)

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

        # NOTE: The agent itself is considered to be a MAX agent
        max_val = self.min_val  # Best reward for current situation under limited moves, initialized with a small enough value
        max_index = 0  # index of the best move
        moves = list(board.get_legal_actions(self.color))  # list of possible moves
        for index, move in enumerate(moves):
            alpha = self.min_val  # Initialize alpha with a small enough value
            beta = self.max_val  # Initialize beta with a big enough value
            flipped = board._move(move, self.color)  # Do the move and remember the pieces got flipped for backpropagation
            val = -self.alpha_beta(board, alpha, beta, self.color, self.depth)  # calculate best reward after making this move
            board.backpropagation(move, flipped, self.color)  # Do backpropagation to restore board state
            max_val = max(val, max_val)  # Update max value
            max_index = index  # Update index of max value

        return moves[max_index]

    def pos_color(self, color):
        return "X" if color is "O" else "O"  # get the color of your opponent

    def sign_color(self, color):
        return 1 if color is self.color else -1  # get the sign of a particular color, positive if equal to self.color
    def value_color(self, color):
        return 1 if color is self.color else (-1 if color is self.pos_color(color) else 0)

    def evaluate(self, board, color):
        result = 0
        for i in range(8):
            for j in range(8):
                result += self.value_color(board._board[i][j]) * self.weight[i][j]
        return self.sign_color(color) * result

    def alpha_beta(self, board, alpha, beta, color, depth, step):
        """
        Calculates the maximum ultimate state income (or after some amount of searchs)
        :param board: Current chess board state (might been already updated)
        :param alpha: The alpha in alpha-beta search, marking the already calculated best option for MAX (previous node on tree)
        :param alpha: The beta in alpha-beta search, marking the already calculated best option for MIN (previous node on tree)
        :param board: Your color for counting reward or getting possible moves
        :param depth: The remaining search depth left to be dealt with
        :return: max_val the maximum reward when considering this node as a MAX node

        NOTE: this method is used for both a MAX node and a MIN node by 
              1. Reversing the color
              2. Exechanging values of alpha and beta after reversing their signs
              so that both can be regarded as a MAX node
              And it reaches termination when alpha is bigger than beta
                    If the value of your children node is smaller than this "alpha" (you're a MIN node), you'll choose this one and
                    the rest of your children can be neglected since they won't be chosen unless smaller than this previously mentioned "value"
                    and your parent node already have a choice better than (bigger than) this "value" ("alpha").
                    If the value of your children node is bigger than this "beta" (you're a MAX node), you'll choose this one and
                    the rest of your children can be neglected since they won't be chosen unless bigger than this previously mentioned "value"
                    and your parent node already have a choice better than (smaller than) this "value" ("beta").
        """
        max_val = self.min_val  # Initializing the return value to be a small enough number, defined in init
        moves = list(board.get_legal_actions(color))  # Get possible moves of current color for future reference
        pos_moves = list(board.get_legal_actions(self.pos_color(color)))  # Get possible moves of the opponent's color for future reference

        # TERMINATION POINT
        if depth <= 0:
            return self.evaluate(board, color)  # Maximum depth is reached
        if len(moves) is 0:
            if len(pos_moves) is 0:
                return self.evaluate(board, color)  # Game is finished
            # POSSIBLE RECURSION
            return -self.alpha_beta(board, -beta, -alpha, self.pos_color(color), depth, step)  # Cannot move, wait for opponent's reaction
        for move in moves:
            flipped = board._move(move, color)  # Save flipped pieces for backpropation
            # POSSIBLE RECURSION
            val = -self.alpha_beta(board, -beta, -alpha, self.pos_color(color), depth - 1, step)  # Move one depth forward
            board.backpropagation(move, flipped, color)  # Restore board state
            max_val = max(max_val, val)
            if max_val >= beta:
                # print("Pruning")
                return max_val
            alpha = max(max_val, alpha)
        return max_val
