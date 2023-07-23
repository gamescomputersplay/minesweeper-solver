''' SImulation to see what is the highest mine density
in which simulator could get at least one win
'''

import random
import time

from tqdm import tqdm

import minesweeper_game as mg
import minesweeper_solver as ms

# Experiment settings:
# Game field size
SIZE = (8, 8)
# Starting number of mines
STARTING_MINES = 20
# Wins to have
WINS_TARGET = 10

# Increase number of mines for the next test
# Decrease if False
NEXT_INCREASE = True

# Max number of attempts until give up
MAX_ATTEMPTS = 100000000


def one_game(solver, settings, seed):
    ''' Playing one game with seed "seed" and setting "settings"
    return True for victory, False for defeat
    '''

    # Initiate the game
    game = mg.MinesweeperGame(settings, seed)

    while game.status == mg.STATUS_ALIVE:

        start = time.time()

        safe, mines = solver.solve(game.uncovered,
                                    deterministic=True,
                                    optimize_for_speed=True)

        game.make_a_move(safe, mines)

        # If one move takes more than 10 seconds, give up
        if time.time() - start > 1:
            return False

    # Game over
    # return True if game status is WON
    if game.status == mg.STATUS_WON:
        return True
    return False
    
def main():
    ''' The experiment itself
    '''
    # Number of mines to have in each iteration
    # will be increased indefinitely
    mines = STARTING_MINES

    while True:

        # Create GameSettings with the required number of mines
        settings = mg.GameSettings(SIZE, mines)

        current_wins = 0

        # Crete solver with required settings
        solver = ms.MinesweeperSolver(settings)

        iterator = tqdm(range(MAX_ATTEMPTS), leave=False, smoothing=0)
        iterator.set_description(f"Size: {SIZE}, Mines: {mines}, {current_wins} win")
        for attempt in iterator:

            # Create a random seed
            seed = random.random()
            result = one_game(solver, settings, seed)
            if result:

                current_wins += 1
                iterator.set_description(f"Size: {SIZE}, Mines: {mines}, {current_wins} win")
                if current_wins == WINS_TARGET:
                    iterator.close()
                    print(f"Mines: {mines}: Wins: {current_wins}, Attempt {attempt+1}, seed {seed}")
                    break

        else:
            print("Reached MAX_ATTEMPTS, no luck")
            break

        # Mine count for the next experiment
        if NEXT_INCREASE:
            mines += 1
        else:
            mines -= 1

if __name__ == "__main__":
    main()
