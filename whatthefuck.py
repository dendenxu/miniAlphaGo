import numpy as np


class AIPlayer:
    '''An AIPlayer that gives moves according to current board status'''

    def __init__(self, color, big_val=1e10, small_val=-1e10, max_depth=4, max_width=12):
        '''
        :param color: Defining the color of the current player, use 'X' or 'O'
        :param big_val: A value big enough for beta initialization
        :param small_val: A value small enough for alpha initialization
        :param max_depth: Max search depth of game tree
        :param max_width: Max search width of game tree, not used in practice
        '''
        self.color = color
        self.oppo_color = "X" if color is "O" else "O"  # Precomputing opponent's color for performance
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

        _, result = self.alpha_beta(board, self.small_val, self.big_val, self.color, self.depth, board.count("X") + board.count("O"))

        print(_, result)
        # ------------------------------------------------------------------------
        return result

    def evaluate(self, board, color, oppo_color):
        """
        Evaluate current situation by weights and stability
        Note that we've try other method than passing the opponent's color as a parameter
        However, since computing that in function alpha_beta is unavoidable, we decide not to repeat the computation
        :param board: Current gaming board to evaluate on
        :param color: Current color
        :param oppo_color: Opponent's color
        """
        # Get current board from the gaming board by a 3-value representation form
        _board = np.asarray([[1 if (piece is color) else (-1 if piece is oppo_color else 0)
                              for piece in line] for line in board._board])
        # Separately arranging the board by segmentation

        # Use vectorized method to compute stability

        # Compute weighted evaluation using vectorized method
        _board *= self.weight
        _board = np.sum(_board)

        # Return the evaluation and cut current move if it leads to failure
        return _board

    def alpha_beta(self, board, alpha, beta, color, depth, step):
        """
        Get the best move under current situation by recursion
        Note that the alpha_beta function always treats current node as MAX node (try to get the maximum reward) and it passes the reward value correctly by reversing the sign of the returned value every time a recursion is to be initiated.
        This technique will significantly reduce possible bugs and maintain a clean code for both MAX and MIN node
        :param board: Current board, need to be reserved after manipulation
        :param alpha: Alpha value that indicates whether we should explore some move for a MIN node
        :param beta: Beta value that indicates whether we should explore some move for a MAX node
        :param color: Current player's color, used in evaluation
        :param depth: Current exploration depth, goes down during recursion, initialized as self.max_depth
        :param step: Current step taken from the beginning of computing an action, used to determine the phase of current game
        :return: max_val the maximum reward of the children of current node
        :return: action the action that generates the maximum reward
        """
        action = None  # Initialize current action for returning
        oppo_color = "X" if color is "O" else "O"  # Pre-compute the opponents color for performance
        max_val = self.small_val  # Initialize the max reward for current situation
        moves = list(board.get_legal_actions(color))  # Get possible moves from calling the board API
        global_depth = step + self.depth - depth  # Compute current steps taken since the beginning of game
        oppo_moves = list(board.get_legal_actions(oppo_color))  # Get opponent's possible move for computing mobility


        # No possible move for current color
        if len(moves) is 0:
            # No possible move for current opponent
            if len(oppo_moves) is 0:
                # Compute mobility
                # Return the evaluation and the action is currently None since no move is to be made
                return self.evaluate(board, color, oppo_color), action
            # If you cannot move, you can only wait till your opponent makes you able to, action is also none
            # Note that we've reversed the sign passed back from the recursion call so that the evaluation value is to current color
            # And we've also reversed alpha, beta since every call is treated as a MAX node
            return -self.alpha_beta(board, -beta, -alpha, oppo_color, depth, step)[0], action

        # If maximum depth is already reached, we should just return the result
        if depth <= 0:
            # Compute mobility
            # You may have notice that mobility is used twice and we computed it seperately
            # This is because you should never use mobility during the expansion of the tree, so pre-computing it might be a waste

            # If already in the last half of the game, we return the normal evaluation which is mobility+stability and weighted board
            return self.evaluate(board, color, oppo_color), action

        # If no terminal state is encountered, we should just enumerate on possible moves computed above
        for move in moves:
            flipped = board._move(move, color)  # Make a move
            val = -self.alpha_beta(board, -beta, -alpha, oppo_color, depth - 1, step)[0]  # Recursively compute the reward
            board.backpropagation(move, flipped, color)  # Reverse the change made to gaming board for the next enumeration
            # Update current maximum reward value
            if val > max_val:
                max_val = val
                action = move
                # Update current alpha value for alpha-beta pruning
                if max_val > alpha:
                    if max_val >= beta:
                        action = move
                        return max_val, action
                    # Update
                    alpha = max_val

        return max_val, action

