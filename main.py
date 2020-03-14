if __name__ == '__main__':
    # 导入黑白棋文件
    from game import Game
    from RandomPlayer import RandomPlayer
    from HumanPlayer import HumanPlayer
    import AIPlayer
    import SimpleAIPlayer
    import MultiCore
    import SingleCore
    import wyq

    max_depth_black = 1
    max_depth_white = 6
    for i in range(1, 6):
        # 人类玩家黑棋初始化
        # black_player = HumanPlayer("X")
        # black_player = RandomPlayer("X")
        # black_player = SimpleAIPlayer.AIPlayer("X", max_depth=max_depth_black)
        # black_player = AIPlayer.AIPlayer("X", max_depth=3)
        black_player = wyq.AIPlayer("X", max_depth=i)
        # black_player = SingleCore.AIPlayer("X", max_depth=max_depth_black)

        # AI 玩家 白棋初始化
        white_player = SingleCore.AIPlayer("O", max_depth=max_depth_white)
        # white_player = wyq.AIPlayer("O", max_depth=max_depth_white)
        # white_player = SimpleAIPlayer.AIPlayer("O")
        # white_player = RandomPlayer("O")
        # 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
        game = Game(black_player, white_player)

        # 开始下棋
        print("Current max depth: {} : {}".format(i, max_depth_white))
        winner, diff = game.run_quite()
        # again = input("兄贵，再试一次？（狗头）：yes/no/change max_depth: ")
        # if again[0] == 'N' or again[0] == 'n':
        #     break
        # if again[0] == 'C' or again[0] == "c":
        #     max_depth_black = int(input("Max depth for black: "))
        #     max_depth_white = int(input("Max depth for white: "))
