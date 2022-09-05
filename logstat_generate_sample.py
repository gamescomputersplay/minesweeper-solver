''' Generate sample boards, to compare statistics with clones' boards
'''

import minesweeper_game as mg

def main():
    ''' Generate boards
    '''
    settings = mg.GAME_EXPERT
    games = 2500
    for _ in range(games):
        game = mg.MinesweeperGame(settings=settings)
        mg.log_field(game.field, log_file_name="log_control.log")

if __name__ == "__main__":
    main()
