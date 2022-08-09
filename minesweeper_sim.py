''' Simulator that plays multiple minesweeepr games
'''

import time
import math

import minesweeper_game as ms
import minesweeper_solver as msolve


class MinesweeperSim:
    ''' Methods to handle minesweeper game simulation
    '''

    def __init__(self, runs, settings):
        self.runs = runs
        self.settings = settings

        # List to store the results
        self.results = []

        # Run the simulation (with timing)
        start_time = time.time()
        self.run()
        self.spent_time = time.time() - start_time

    def one_game(self, verbose=False):
        ''' Playing one game.
        Verbose would show board for every move
        '''

        game = ms.MinesweeperGame(self.settings)
        solver = msolve.MinesweeperSolver(self.settings)

        while game.status == ms.STATUS_ALIVE:

            safe, mines = solver.solve(game.uncovered)
            if verbose:
                print(f"Player: safe={safe}, mines={mines}")
            game.make_a_move(safe, mines)
            if verbose:
                print(game)

        if verbose:
            print(f"Result: {ms.STATUS_MESSAGES[game.status]}")

        return game.status

    def run(self):
        ''' Running the simulation
        '''
        for _ in range(self.runs):
            self.results.append(self.one_game())
        return self.winrate()

    def winrate(self):
        ''' Return current winrate
        '''
        return self.results.count(ms.STATUS_WON) / len(self.results)

    def margin_of_error(self):
        ''' Current margin of error (95% confidence).
        In percetange points (+- that many %).
        '''
        runs = len(self.results)
        winrate = self.winrate()
        z_parameter = 1.95  # Corresponds to 95% confidence
        return z_parameter * math.sqrt(winrate * (1 - winrate) / runs)

    def display_stats(self):
        ''' Print out the stats of teh simulation
        '''

        if not self.results:
            return "No data to display"

        output = ""

        winrate = self.winrate()
        output += f"Win rate: {winrate:.1%}±{self.margin_of_error():.1%}\n"
        output += f"Total time: {self.spent_time:.0f}s, "
        output += f"Time per game: {self.spent_time / self.runs:.2f}s\n"

        return output

    def __str__(self):
        ''' Show parameters of teh simulation
        '''

        output = f"Settings: {self.settings}, " + \
                 f"Runs: {self.runs}"
        return output


def main():
    ''' Run a sample simulation
    '''
    for settings in (ms.GAME_BEGINNER, ms.GAME_INTERMEDIATE, ms.GAME_EXPERT,
                     ms.GAME_3D_EASY, ms.GAME_3D_MEDUIM, ms.GAME_3D_HARD,
                     ms.GAME_4D_EASY):
        simulation = MinesweeperSim(100, settings)
        print(simulation)
        print(simulation.display_stats())


if __name__ == "__main__":
    main()
