if __name__ == '__main__':
    # 导入黑白棋文件
    from game import Game
    from RandomPlayer import RandomPlayer
    from HumanPlayer import HumanPlayer
    import AIPlayer
    import SimpleAIPlayer
    import MultiCore
    import SingleCore
    # 人类玩家黑棋初始化
    # black_player = HumanPlayer("X")
    # black_player = RandomPlayer("X")
    black_player = SimpleAIPlayer.AIPlayer("X", max_depth=3)
    # black_player = AIPlayer.AIPlayer("X", max_depth=3)

    # AI 玩家 白棋初始化
    white_player = AIPlayer.AIPlayer("O", max_depth=3)
    # white_player = SimpleAIPlayer.AIPlayer("O")
    # white_player = RandomPlayer("O")

    while True:
        # 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
        game = Game(black_player, white_player)

        # 开始下棋
        game.run()
        again = input("兄贵，再试一次？（狗头）：yes/no: ")
        if again[0] == 'N' or again[0] == 'n':
            break