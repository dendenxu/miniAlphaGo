import datetime
import math
import random
from copy import deepcopy
from board import Board


def terminal_test(board):
    b_list = list(board.get_legal_actions('X'))
    w_list = list(board.get_legal_actions('O'))

    is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

    return is_over


def switch_player(color):
    return 'O' if color == 'X' else 'X'


def IsOnCorner(x, y):
    # Returns True if the position is in one of the four corners.
    return (x == 0 and y == 0) or (x == 7 and y == 0) or (x == 0 and y == 7) or (x == 7 and y == 7)


class TreeNode:
    """
    MCTS树节点
    """

    def __init__(self, state, parent, action, color):
        self.state = deepcopy(state)  # the state the node represents
        self.parent = parent
        self.children = []
        self.action = action  # the action that leading to the node
        self.color = color  # finish the action's player's color
        self.N = 0  # the number of times node has been visted
        self.Q = 0  # the total reward of the action leading to the node

    def all_actions(self):
        board = Board()
        board._board = deepcopy(self.state)
        next_player = switch_player(self.color)
        action_list = list(board.get_legal_actions(next_player))
        return action_list

    def is_full_expanded(self):
        return len(self.children) == len(self.all_actions())

    def is_leaf(self):
        return len(self.children) == 0

    def avg_reward(self):
        return self.Q / self.N

    def b(self, c):
        return (c * math.sqrt(2 * math.log(self.parent.N) / self.N))

    def get_valid_actions(self):
        valid_actions = deepcopy(self.all_actions())
        for child in self.children:
            valid_actions.remove(child.action)
        return valid_actions

    def is_terminal(self):
        return len(self.all_actions()) == 0


class AIPlayer:
    """
    MCTS
    """

    def __init__(self, color, c=1/math.sqrt(2), search_time_interval=5):
        self.color = color
        self.op_color = 'O' if self.color == 'X' else 'X'
        self.root = TreeNode(None, None, None, self.op_color)
        self.c = c
        self.calculation_time = datetime.timedelta(
            seconds=search_time_interval)

    def UCTSearch(self, board):
        input_state = board._board
        if len(self.root.children) != 0:
            flag = False  # the flag whether the state is the root's child
            for child in self.root.children:
                if input_state == child.state:
                    self.root = child
                    flag = True
                    break
            if not flag:  # if the state is not in the children
                self.root = TreeNode(board._board, None, None, self.op_color)
        else:
            self.root = TreeNode(board._board, None, None, self.op_color)

        begin, turn = datetime.datetime.utcnow(), 0
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            turn = turn + 1
            node = self.tree_policy(self.root)
            delta = self.default_policy(node)
            self.back_propagate(node, delta)

        self.root = self.best_child(self.root)
        a = self.root.action
        print("Simulate %d runs" % turn)
        return a

    def expand(self, node):

        board = Board()
        board._board = deepcopy(node.state)

        player = switch_player(node.color)

        valid_action = node.get_valid_actions()

        bestVal = -math.inf
        best_node = None
        for action in valid_action:
            flipped_pos = board._move(action, player)
            new_node = TreeNode(board._board, node, action, player)
            x, y = board.board_num(action)
            if IsOnCorner(x, y):
                node.children.append(new_node)
                return new_node
            val = len(flipped_pos)
            if val > bestVal:
                best_node = new_node
                bestVal = val
            board.backpropagation(action, flipped_pos, player)

        node.children.append(best_node)
        return best_node

    def best_child(self, node):
        if len(node.children) > 0:
            if node.color == self.op_color:
                return max(node.children, key=lambda x: (x.avg_reward() + x.b(self.c)))
            else:
                return max(node.children, key=lambda x: (1 - x.avg_reward() + x.b(self.c)))
        else:
            return None

    def tree_policy(self, node):
        board = Board()
        board._board = deepcopy(node.state)
        while not node.is_terminal():  # node is nonterminal
            if not node.is_full_expanded():  # not fully expanded
                return self.expand(node)
            elif not node.is_leaf():
                node = self.best_child(node)  # child is the oppenent's chess
        return node

    def default_policy(self, node):
        board = Board()
        board._board = deepcopy(node.state)
        player = node.color
        while not terminal_test(board):
            player = switch_player(player)
            action_list = list(board.get_legal_actions(player))
            if len(action_list) == 0:
                continue
            action = random.choice(action_list)
            board._move(action, player)
        if (board.count(self.color) - board.count(self.op_color)) > 0:
            return 1
        elif (board.count(self.color) - board.count(self.op_color)) < 0:
            return -1
        else:
            return 0

    def back_propagate(self, node, delta):
        while node is not None:
            node.N += 1
            node.Q += delta
            node = node.parent

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
        action_list = list(board.get_legal_actions(self.color))
        if len(action_list) == 0:
            return None

        action = self.UCTSearch(board)

        return action
