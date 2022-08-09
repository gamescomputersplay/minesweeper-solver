''' Class for the solver of minesweeper game.
Game mechanics and some helper functions are taken from
minesweeper-game.py
'''

import random
import minesweeper_game as ms


class MineGroup:
    ''' A minegroup is a set of cells that are know
    to have a certian number of mines.
    The basic element for many solving methods
    '''

    def __init__(self, cells, mines):
        ''' Cells in questions. Number of mines that those cells have.
        '''
        # Use set rather than list as it speeds up some calculations
        self.cells = set(cells)
        self.mines = mines

    def is_all_safe(self):
        ''' If group has 0 mines, it is safe
        '''
        return self.mines == 0

    def is_all_mines(self):
        ''' If group has as many cellas mines - they are all mines
        '''
        return len(self.cells) == self.mines

    def __str__(self):
        ''' List cells and mines
        '''
        out = "Cell(s) "
        out += ",".join([str(cell) for cell in self.cells])
        out += f" have {self.mines} mines"
        return out


class MinesweeperSolver:
    ''' Methods related to solving minesweeper game. '''

    def __init__(self, settings=ms.GAME_BEGINNER):
        ''' Initiate the solver. Only requred game settings
        '''

        # Shape, a tuple of dimension sizes
        self.shape = settings.shape
        # Number of total mines in the game
        self.total_mines = settings.mines

        # Initiate helper (itiration through all cells, neighbours etc)
        self.helper = ms.MinesweeperHelper(self.shape)

        # Placeholder for thej field. Will be populated by self.solve()
        self.field = None

        # Placeholder for all groups. Recalculated for each solver run
        self.groups = []

    def get_all_covered(self):
        ''' Return the list of all covered cells
        '''
        all_covered = []
        for cell in self.helper.iterate_over_all_cells():
            if self.field[cell] == ms.CELL_COVERED:
                all_covered.append(cell)
        return all_covered

    @staticmethod
    def pick_a_random_cell(cells):
        '''Pick a random cell out of the list of cells.
        (Either for testing or when we are reduced to guessing)
        '''
        return random.choice(cells)

    def generate_groups(self):
        '''Populate self.group with all found cell groups
        '''
        self.groups = []

        # Go over all cells and find all the "Numbered ones"
        for cell in self.helper.iterate_over_all_cells():
            if self.field[cell] > 0:

                # For them we'll need to know two things:
                # What are the uncovered cells around it
                covered_neighbours = []
                # And how many "Acive" (that is, minus marked)
                # mines are still there
                active_mines = self.field[cell]

                # Go through the neighbours
                for neighbour in self.helper.cell_surroundings(cell):
                    # Collect all covered cells
                    if self.field[neighbour] == ms.CELL_COVERED:
                        covered_neighbours.append(neighbour)
                    # Substruct all marked mines
                    if self.field[neighbour] == ms.CELL_MINE:
                        active_mines -= 1

                # If the list of covered cells is not empty:
                # store it in the self.groups
                if covered_neighbours:
                    new_group = MineGroup(covered_neighbours, active_mines)
                    self.groups.append(new_group)

    def naive_method(self):
        ''' Method #1. Naive
        Check if there are all safe or all mines groups.
        Return safe and mines found in those groups
        '''
        safe, mines = [], []
        for group in self.groups:
            if group.is_all_safe():
                safe.extend(list(group.cells))
            if group.is_all_mines():
                mines.extend(list(group.cells))
        return list(set(safe)), list(set(mines))

    def solve(self, field):
        ''' Main solving function.
        Go through various solving methods and return safe and mines lists
        as long as any of the methods return results
        In: the field (what has been uncovered so far).
        Out: two lists: [safe cells], [mine cells]
        '''
        # Store field as an instance variable
        self.field = field

        covered_cells = self.get_all_covered()

        # if it is a very first move: pick a random cell
        if self.helper.are_all_covered(self.field):
            return [self.pick_a_random_cell(covered_cells), ], None

        # Generate groups (main data for basic solving methods)
        self.generate_groups()

        # Try naive method. If it worked, return safe and mines lists
        safe, mines = self.naive_method()
        if safe or mines:
            return safe, mines

        # If there is nothing we can do: return a random cell
        return [self.pick_a_random_cell(covered_cells), ], None


def main():
    ''' Test the solver on a simple game
    '''

    settings = ms.GAME_TEST

    game = ms.MinesweeperGame(settings)
    solver = MinesweeperSolver(settings)

    while game.status == ms.STATUS_ALIVE:

        safe, mines = solver.solve(game.uncovered)
        print(safe, mines)
        game.make_a_move(safe, mines)
        print(game)

    print(game.status)


if __name__ == "__main__":
    main()
