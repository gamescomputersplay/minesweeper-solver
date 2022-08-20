''' Simulator program that plays multiple minesweeper games.
The game is from minesweeper_game, the solver is from minesweeper_solver
'''

import time
import math
import random

# progress bar
from tqdm import tqdm
# text table
from texttable import Texttable

import minesweeper_game as ms
import minesweeper_solver as ms_solver

# Whether to display progress bar for a simulation
# May have issues on some IDEs
USE_PROGRESS_BAR = True
SHOW_METHODS_STAT = True
SHOW_PROBABILITY_STAT = True


class SolverStat:
    ''' Class to collect and display statistics about methods, used by solver
    '''

    def __init__(self):
        # main repository for the data. Will be a dict like
        # {"Naive": 100, "Group":200}
        self.data = {}
        # Games counter
        self.games = 0
        # Wins counter
        self.wins = 0

        # Data about probability. Dict like:
        # {"Probability_method": [0.5, 0.1, -0.2]}
        # Where negative number means it resulted in death
        self.probability_data = {}

        # Last probability method that was used (it will be blamed
        # if the game is lost)
        self.last_probability_method = None

    def add_move(self, last_move_info, safe, mines):
        ''' Add move information
        '''
        method, prob_method, chance = last_move_info

        # Count clicks as the sum of safe and mines
        count = 0
        for clicks in (safe, mines):
            if clicks:
                count += len(clicks)

        # Add the data about the method that was used
        self.data[method] = self.data.get(method, 0) + count

        # Add the data about the probability method that was used
        if method == "Probability":
            # Create a list to hold info about this method
            if prob_method not in self.probability_data:
                self.probability_data[prob_method] = []

            # Add the latest chance
            self.probability_data[prob_method].append(chance)
            # Remember which was the last probability method we used
            # (we will blame it wor the lost)
            self.last_probability_method = prob_method

    def add_game(self, result):
        ''' Add game information
        '''
        self.games += 1
        if result == ms.STATUS_WON:
            self.wins += 1
        # If we lost
        elif result == ms.STATUS_DEAD:
            # Than blame the latest probability method
            # Mark it by inverting the chance value
            self.probability_data[self.last_probability_method][-1] = \
                -self.probability_data[self.last_probability_method][-1]

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

    def table_by_method(self):
        '''Generate a text table with information about method being used
        '''
        # Sum of all clicked cells
        total_cells = sum(self.data.values())
        # Table to display
        table_data = [["Method", "Total", "Per game", "%"]]
        for method, count in self.data.items():
            table_data.append([method,
                               count,
                               count / self.games,
                               count * 100 / total_cells])

        # Generate table with texttable module
        table = Texttable()
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(['t', 'i', 'f', 'f'])
        table.set_cols_align(["l", "r", "r", "r"])
        table.add_rows(table_data)
        return "Clicks by solving method:\n" + table.draw() + "\n\n"

    def table_by_probability_method(self):
        ''' Generate a table with results of various random methods.
        Stats are presented by "buckets", in two ways: exactly the value
        and a range between values.
        '''

        def display_probability(guesses, loses):
            '''The string to write to the probability table.
            Also, some sanitization.
            '''
            guesses_str = f"{guesses}" if guesses < 1000 \
                          else f"{guesses//1000}k"
            if 0 < guesses > threshold:
                return f"{loses / guesses:.3f} ({guesses_str})"
            return ""

        def fill_the_bucket(exactly=None, low=None, high=None):
            '''Return the stats for particular "Bucket". Bucket can be
            exactly that probability or a range of probability
            Return the dict: {"Prob method": (actual probability, count)}
            '''
            bucket = []

            # Counter for overall result
            total_guesses = 0
            total_loses = 0

            # We go through all the probability data by method
            for _, method_data in self.probability_data.items():

                # We'll count how many guesses there were
                # Adn how many resulted in death
                methods_guesses = 0
                methods_loses = 0

                for probability in method_data:
                    # Check if the probability falls into the bucket,
                    # be it "exactly" or a range
                    if exactly is not None and abs(probability) == exactly or \
                       low is not None and low < abs(probability) < high:
                        methods_guesses += 1
                        total_guesses += 1
                        if probability < 0:
                            methods_loses += 1
                            total_loses += 1

                # Store the result in the final dict
                bucket.append(display_probability(methods_guesses,
                                                  methods_loses))

            # Store the total result in the final dict
            bucket.append(display_probability(total_guesses,
                                              total_loses))

            return bucket

        # Sizeof a bucket to group results in
        buckets_edges = [0, .1, .25, 1/3, .5, 1]
        # Only display data that has at least this many data points
        threshold = 10

        table_data = [["Predicted"] + list(self.probability_data) + ["Total"]]

        # This funky loop and the conditions below allows us to generate
        # 2 type of buckets: exactly that value and range between two values
        for i in range(1, (len(buckets_edges) - 1) * 2):

            if i % 2 == 0:
                # The exactly bucket
                exactly = buckets_edges[i // 2]
                bucket = [f" {exactly:.2f}"] + \
                    fill_the_bucket(exactly=exactly)
            else:
                # The "range" bucket
                low = buckets_edges[(i - 1) // 2]
                high = buckets_edges[(i + 1) // 2]
                bucket = [f"{low:.2f}-{high:.2f}"] + \
                    fill_the_bucket(low=low, high=high)

            table_data.append(bucket)

        # Generate table with texttable module
        table = Texttable()
        table.set_deco(Texttable.HEADER)
        table.set_cols_align(["r"] + ["l"] * (len(table_data[0]) - 1))
        table.add_rows(table_data)
        return "Probability accuracy control:\n" + table.draw() + "\n\n"

    def __str__(self):
        ''' Display the stats
        '''
        win_rate = self.win_rate()
        output = ""
        if SHOW_METHODS_STAT:
            output += self.table_by_method()
        if SHOW_PROBABILITY_STAT:
            output += self.table_by_probability_method()
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
        # Add game results to teh statistics
        self.solver_stat.add_game(game.status)

        # Notify if deterministic method resulted in death
        # This should not happen though
        if game.status == ms.STATUS_DEAD and \
           solver.last_move_info[0] != "Probability":
            print(f"Warning: death by method '{solver.last_move_info[0]}'")

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

        print(timing_info, "\n")
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
