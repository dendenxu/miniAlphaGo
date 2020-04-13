if __name__ == '__main__':
    from game import Game
    from HumanPlayer import HumanPlayer
    from RandomPlayer import RandomPlayer
    import MultiCore
    import SingleCore
    import really_simple
    import cwy_mtkl

    while True:
        black_player = RandomPlayer("X")
        white_player = MultiCore.AIPlayer("O", max_depth=6)
        # init_board = None
        init_board = [
            ['.', '.', 'X', 'X', 'X', 'X', '.', '.'],
            ['.', '.', 'X', 'X', 'X', 'X', '.', '.'],
            ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            ['X', 'X', 'X', 'O', 'X', 'X', 'X', 'X'],
            ['X', 'X', 'X', '.', 'X', 'X', 'X', 'X'],
            ['.', '.', '.', 'O', 'O', 'O', '.', '.'],
            ['.', '.', '.', '.', '.', 'O', '.', '.'],
        ]
        game = Game(black_player, white_player, init_board)
        game.run()
        again = input("再试一次？：yes/no: ")
        if again[0] == 'N' or again[0] == 'n':
            break
