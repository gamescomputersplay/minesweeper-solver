''' Simulator that plays multiple minesweeepr games
'''

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

        # Run the simulation
        self.run()

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

    def display_stats(self):
        ''' Print out the stats of teh simulation
        '''

        if not self.results:
            return "No data to display"

        output = ""

        winrate = self.results.count(2) / len(self.results)
        output += f"Win rate: {winrate:.1%}\n"

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
                     ms.GAME_4D):
        simulation = MinesweeperSim(100, settings)
        print(simulation)
        print(simulation.display_stats())


if __name__ == "__main__":
    main()
