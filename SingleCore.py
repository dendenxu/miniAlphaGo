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
        self.weight = np.asarray([[150, -80, 10, 10, 10, 10, -80, 150],  # weight matrix of board position
                                  [-80, -90,  5,  5,  5,  5, -90, -80],
                                  [ 10,   5,  1,  1,  1,  1,   5,  10],
                                  [ 10,   5,  1,  1,  1,  1,   5,  10],
                                  [ 10,   5,  1,  1,  1,  1,   5,  10],
                                  [ 10,   5,  1,  1,  1,  1,   5,  10],
                                  [-80, -90,  5,  5,  5,  5, -90, -80],
                                  [150, -80, 10, 10, 10, 10, -80, 150]])
        # self.weight = np.asarray([[100, -5, 10, 5, 5, 10, -5, 100],  # weight matrix of board position
        #                           [-5, -45, 1, 1, 1, 1, -45, -5],
        #                           [10, 1, 3, 2, 2, 3, 1, 10],
        #                           [5, 1, 2, 1, 1, 2, 1, 5],
        #                           [5, 1, 2, 1, 1, 2, 1, 5],
        #                           [10, 1, 3, 2, 2, 3, 1, 10],
        #                           [-5, -45, 1, 1, 1, 1, -45, -5],
        #                           [100, -5, 10, 5, 5, 10, -5, 100]])
        self.factor = 50  # How much mobility and stability affects the evaluation of current board

        # History table for evaluating table
        # axis=0: color of history table
        # axis=0: current moves step since the beginning of the game
        self.history = np.tile(np.arange(64), 128).reshape((2, 64, 64))

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

        # print(result)
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
        sep_board = np.stack(((_board == 1).astype(int), np.negative((_board == -1).astype(int))))
        # Use vectorized method to compute stability
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

        # Compute weighted evaluation using vectorized method
        _board *= self.weight
        _board = np.sum(_board)
        _board += stability * self.factor

        # Return the evaluation and cut current move if it leads to failure
        return _board if np.sum(sep_board[0, :, :]) else self.small_val

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

        goods = ["A1", "A8", "H1", "H8"]
        if depth == self.depth:
            for good in goods:
                if good in moves:
                    return 0, good

        # bads = ["A2", "B1", "B2", "A7", "B7", "B8", "G1", "H2", "G2", "G7", "G8", "H7"]
        # if depth == self.depth:
        #     for bad in bads:
        #         if bad in moves:


        # No possible move for current color
        if len(moves) is 0:
            # No possible move for current opponent
            if len(oppo_moves) is 0:
                # Compute mobility
                mobility = (len(moves) - len(oppo_moves)) * self.factor
                # Return the evaluation and the action is currently None since no move is to be made
                return self.evaluate(board, color, oppo_color) + mobility, action
            # If you cannot move, you can only wait till your opponent makes you able to, action is also none
            # Note that we've reversed the sign passed back from the recursion call so that the evaluation value is to current color
            # And we've also reversed alpha, beta since every call is treated as a MAX node
            return -self.alpha_beta(board, -beta, -alpha, oppo_color, depth, step)[0], action

        # If maximum depth is already reached, we should just return the result
        if depth <= 0:
            # Compute mobility
            # You may have notice that mobility is used twice and we computed it seperately
            # This is because you should never use mobility during the expansion of the tree, so pre-computing it might be a waste
            mobility = (len(moves) - len(oppo_moves)) * self.factor
            if global_depth < 16:
                # Try to maintain mobility during the first steps of gaming
                # A strategy called: Disappearing piece
                return mobility, action
            # If already in the last half of the game, we return the normal evaluation which is mobility+stability and weighted board
            return self.evaluate(board, color, oppo_color) + mobility, action

        # Sort our moves according to the history table
        moves = self.history_sort(board, moves, color, global_depth)

        # NOT USED: limit the maximum width to expand on
        # moves = moves[::min(len(moves), self.max_width)]

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
                        # The other children of current node should not be checked anymore
                        # and reward current best move since is leads to an alpha-beta pruning
                        self.reward_move(board, action, color, global_depth, True)
                        return max_val, action
                    # Update
                    alpha = max_val

        # Reward current move since it's not the worst move
        if action is not None:
            # Note that sometimes no action can exist so we check
            self.reward_move(board, action, color, global_depth, False)
        return max_val, action

    def history_sort(self, board, moves, color, depth):
        '''
        Sort possible moves by their previous performance
        :param board: Current state of gaming
        :param moves: The moves involved
        :param color: Current color
        :param depth: How many steps since the beginning of the game
        :return: sorted moves based on history table (previous performance of every move)
        '''
        if depth >= 64:
            return moves
        # get all possible positions for the moves argument we've accepted
        poss = list(map(lambda x: x[0] * 8 + x[1], [board.board_num(move) for move in moves]))
        # numpy grants argument indexing with np.array
        values = self.history[int(color == self.color), depth][poss]
        # perform argsort based on the values we've got
        idx = np.argsort(values)
        # Return the moves in a sorted order
        return np.asarray(moves)[idx]

    def reward_move(self, board, move, color, depth, best):
        '''
        Reward a move by raising its ranking if it leads to pruning or is not the worst move
        :param board: Current state of gaming
        :param moves: The move to be rewarded
        :param color: Current color
        :param depth: How many steps since the beginning of the game
        :param depth: Whether the move to be rewarded is a best move (leads to pruning)
        :return: nothing
        '''
        if depth >= 64:
            return
        # Compute the corresponding position of the move to be rewarded
        x, y = board.board_num(move)
        pos = x * 8 + y
        color = int(color == self.color)

        # Get value
        val = self.history[color, depth, pos]
        other_pos = np.argwhere(self.history[color, depth, :] == val - (val if best else (1 if val else 0)))

        # Exchange position
        self.history[color, depth, other_pos], self.history[color, depth, pos] = \
            self.history[color, depth, pos], self.history[color, depth, other_pos]
