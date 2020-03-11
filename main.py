# 导入黑白棋文件
from game import Game  
from RandomPlayer import RandomPlayer
from HumanPlayer import HumanPlayer
from AIPlayer import AIPlayer
# 人类玩家黑棋初始化
black_player = HumanPlayer("X")
black_player = RandomPlayer("O")

# AI 玩家 白棋初始化
white_player = AIPlayer("O", max_depth=3)
# white_player = RandomPlayer("O")

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)

# 开始下棋
game.run()