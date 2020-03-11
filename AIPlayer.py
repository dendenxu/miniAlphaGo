from board import Board


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
        self.max_val = 1e10
        self.min_val = -1e10
        self.depth = 3

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

        max_val = self.min_val
        max_index = 0
        moves = list(board.get_legal_actions(self.color))
        for index, move in enumerate(moves):
            alpha = self.max_val
            beta = self.min_val
            flipped = board._move(move, self.color)
            val = self.alpha_beta(board, alpha, beta, self.color, self.depth)
            board.backpropagation(move, flipped, self.color)
            max_val = max(val, max_val)
            max_index = index


        return moves[max_index]

    def pos_color(self, color):
        return "X" if color is "O" else "O"  # get the color of your opponent

    def sign_color(self, color):
        return 1 if color is self.color else -1  # get the sign of a particular color, positive if equal to self.color


    def alpha_beta(self, board, alpha, beta, color, depth):
        '''
        Input: board board, alpha α, beta β, side to move color and remain depth depth
        Output: useful maximum of its children notes
        1   max ← −∞;
        2   sign[my color] = 1;
        3   sign[opp color] = − 1;
        4   if depth ≤ 0 then
        5       return sign[color] · Evaluate(board);
        6   if ! CanMove(board, color) then
        7       if ! CanMove(board, my color + opp color − color) then
        8           return sign[color] · Evaluate(board);
        9       return -AlphaBeta(board, − β, − α, my color + opp color − color, depth);
        10  foreach move do
        11      MakeMove(board, move, color);
        13      UnMakeMove(board, move, color);
        12      val ← -AlphaBeta(board, − β, − α, my color + opp color − color, depth − 1);
        14      if val > α then
        15          if val ≥ β then
        16              return val;
        17          α = max(val, α);
        18      max = max(val, max);
        19  return max;
        '''
        max_val = self.min_val
        moves = list(board.get_legal_actions(color))
        pos_moves = list(board.get_legal_actions(self.pos_color(color)))
        if depth <= 0:
            return self.sign_color(color) * (board.count(color) - board.count(self.pos_color(color)))
        if len(moves) is 0:
            if len(pos_moves) is 0:
                return  self.sign_color(color) * (board.count(color) - board.count(self.pos_color(color)))
            return -self.alpha_beta(board, -beta, -alpha, self.pos_color(color), depth)
        for move in moves:
            flipped = board._move(move, color)
            val = -self.alpha_beta(board, -beta, -alpha, self.pos_color(color), depth-1)
            board.backpropagation(move, flipped, color)
            if val > alpha:
                if val < beta:
                    return val
                alpha = max(val, alpha)
            max_val = max(val, max_val)
        return max_val
