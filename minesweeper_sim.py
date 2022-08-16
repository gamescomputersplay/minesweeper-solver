''' Simulator program that plays multiple minesweeper games.
The game is from minesweeper_game, the solver is from minesweeper_solver
'''

import time
import math
import random
from tqdm import tqdm

import minesweeper_game as ms
import minesweeper_solver as ms_solver

# Whether to display progress bar for a simulation
# May have issues on some IDEs
USE_PROGRESS_BAR = True

class SolverStat:
    ''' Class to collect and display statistics about methods, used by solver
    '''

    def __init__(self):
        # main repository for the data. Will be a dict like
        # {"Naive": 100, "Group":200}
        self.data = {}
        self.games = 0
        self.wins = 0

    def add_move(self, last_move_info, safe, mines):
        ''' Add move information
        '''
        method, _, _ = last_move_info

        # Count clicks as the sum of safe and mines
        count = 0
        for clicks in (safe, mines):
            if clicks:
                count += len(clicks)

        self.data[method] = self.data.get(method, 0) + count

    def add_game(self, result):
        ''' Add game information
        '''
        self.games += 1
        if result == ms.STATUS_WON:
            self.wins += 1

    def win_rate(self):
        ''' Return current win_rate
        '''
        if self.games:
            return self.wins / self.games
        return 0

    def margin_of_error(self):
        ''' Current margin of error (95% confidence).
        In percentage points (+- that many %).
        '''
        win_rate = self.win_rate()
        z_parameter = 1.95  # Corresponds to 95% confidence
        return z_parameter * math.sqrt(win_rate * (1 - win_rate) / self.games)

    def __str__(self):
        ''' Display the stats
        '''
        win_rate = self.win_rate()
        output = ""
        output += f"Win rate: {win_rate:.1%}±{self.margin_of_error():.1%}\n"
        return output


class MinesweeperSim:
    ''' Methods to handle minesweeper game simulation
    '''

    def __init__(self, runs, settings, seed=None):
        self.runs = runs
        self.settings = settings

        # Use seed, if seed passed
        if seed is not None:
            random.seed(seed)

        # Generate starting seeds for all games (do it now, so random
        # calls in the solver would not affect it).
        self.game_seeds = [random.random() for _ in range(self.runs)]

        # Placeholder for timing
        self.spent_time = None

        # statistics collector object
        self.solver_stat = SolverStat()

    def one_game(self, verbose=False):
        ''' Playing one game.
        Verbose would print out the board for every move
        '''

        # Start the game, using one of the seeds
        game = ms.MinesweeperGame(self.settings, self.game_seeds.pop())
        solver = ms_solver.MinesweeperSolver(self.settings)

        while game.status == ms.STATUS_ALIVE:

            safe, mines = solver.solve(game.uncovered)
            if verbose:
                print(f"Player: safe={safe}, mines={mines}")
            game.make_a_move(safe, mines)

            # Send all the data into the statistics object
            self.solver_stat.add_move(solver.last_move_info, safe, mines)

            if verbose:
                print(game)

        # Game over
        self.solver_stat.add_game(game.status)
        if verbose:
            print(f"Result: {ms.STATUS_MESSAGES[game.status]}")

        return game.status

    def run(self):
        ''' Running the simulation
        '''
        # Choose if we need to display a progress bar
        if USE_PROGRESS_BAR:
            iterator = tqdm(range(self.runs), leave=False)
        else:
            iterator = range(self.runs)

        # Run the simulation (with timing)
        start_time = time.time()
        for _ in iterator:
            self.one_game()
        self.spent_time = time.time() - start_time

        # Return the overall win rate
        return self.solver_stat.win_rate()

    def print_stats(self):
        ''' Print out the stats of teh simulation
        '''
        timing_info = f"Simulation complete in: {self.spent_time:.0f}s, "
        timing_info += f"Time per game: {self.spent_time / self.runs:.2f}s, "
        timing_info += f"Games per sec: {self.runs / self.spent_time:.2f}s"

        print(timing_info)
        print(self.solver_stat)

    def __str__(self):
        ''' Show parameters of teh simulation
        '''

        output = f"Settings: {self.settings}" + \
                 f"Runs: {self.runs}"
        return output


def main():
    ''' Run a sample simulation
    '''

    # use seed for replicable simulations
    seed = 0

    # games to simulate
    runs = 100

    # All popular minesweeper and multidimensional minesweeper presets
    presets = (ms.GAME_BEGINNER, ms.GAME_INTERMEDIATE, ms.GAME_EXPERT,
               ms.GAME_3D_EASY, ms.GAME_3D_MEDIUM, ms.GAME_3D_HARD,
               ms.GAME_4D_EASY, ms.GAME_4D_MEDIUM, ms.GAME_4D_HARD,
               ms.GAME_4D_HARDER)
    # Only 2D (traditional) minesweeper presets
    presets = (ms.GAME_BEGINNER, ms.GAME_INTERMEDIATE, ms.GAME_EXPERT)
    # Only a small, beginner 2D board (for testing)
    presets = (ms.GAME_BEGINNER, )

    for settings in presets:

        simulation = MinesweeperSim(runs, settings, seed)
        print(simulation)
        simulation.run()
        simulation.print_stats()


if __name__ == "__main__":
    main()
