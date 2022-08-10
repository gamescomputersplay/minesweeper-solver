''' Class for the solver of minesweeper game.
Game mechanics and some helper functions are taken from
minesweeper-game.py
'''

import random
import minesweeper_game as ms


class MineGroup:
    ''' A minegroup is a set of cells that are known
    to have a certian number of mines.
    For example "cell1, cell2, cell3 have exactly 1 mine" or
    "cell4, cell5 have at least 1 mine".
    This is a basic element for Groups and Subgroups solving methods.
    '''

    def __init__(self, cells, mines, group_type="exactly"):
        ''' Cells in questions. Number of mines that those cells have.
        '''
        # Use set rather than list as it speeds up some calculations
        self.cells = set(cells)
        self.mines = mines

        # Group type (exctly, no more than, no fewer than)
        self.group_type = group_type

        # Calculate hash (for dedupliaction)
        self.hash = self.calculate_hash()

    def is_all_safe(self):
        ''' If group has 0 mines, it is safe
        '''
        return self.mines == 0

    def is_all_mines(self):
        ''' If group has as many cellas mines - they are all mines
        '''
        return len(self.cells) == self.mines

    def calculate_hash(self):
        ''' Hash of a group. To check if such group is already in self.group
        '''
        # Prepare data for hashing: sort cells, add self.mines
        for_hash = sorted(list(self.cells)) + [self.mines] + [self.group_type]
        # Make immutable
        for_hash = tuple(for_hash)
        # Return hash
        return hash(for_hash)

    def __str__(self):
        ''' List cells, mines and group type
        '''
        out = "Cell(s) "
        out += ",".join([str(cell) for cell in self.cells])
        out += f" have {self.group_type} {self.mines} mines"
        return out


class AllMineGroups:
    ''' Functions to handle a group of MineGroup object:
    deduplicate them, generate subgroups ("at least" and "no more than")
    '''

    def __init__(self):
        # Hashes for deduplication
        self.hashes = set()
        # List of minegroups
        self.mine_groups = []

    def reset(self):
        ''' Clear the data
        '''
        self.hashes = set()
        self.mine_groups = []

    def add(self, new_group):
        ''' Check if if have this group already.
        If not, add to the list
        '''
        if new_group.hash not in self.hashes:
            self.mine_groups.append(new_group)
            self.hashes.add(new_group.hash)

    def __iter__(self):
        ''' For iterator, use the list of groups
        '''
        return iter(self.mine_groups)

    def generate_subgroup_at_least(self):
        ''' Generate "group has at least X mines" subgroup. Add them to groups.
        Done by removing a cell and a mine, in whichever groups it is possible.
        For example, if cell1, cell2, cell3 have 2 mines, than  cell1 and cell2
        have at least 1 mine, so are cell2 and cell3, cell3 and cell1.
        '''
        for group in self:

            # Doing it for cells in the middle of nowhere would take a lot of
            # resources but virtually never useful.
            if len(group.cells) == 8:
                continue

            # Group should have 3>2 cells and > 1 mines
            # And  be either "exactly" or "at least" group
            if len(group.cells) > 2 and group.mines > 1 and \
               group.group_type in ("exactly", "at least"):

                # Remove each cell and one mine - those  are your new subgroups
                for cell in group.cells:
                    new_subgroup = MineGroup(group.cells.difference({cell}),
                                             group.mines - 1, "at least")
                    # They will be added to the end of the list, so they
                    # in turn will be broken down, if needed
                    self.add(new_subgroup)

    def generate_subgroup_no_more_than(self):
        ''' Generate a second type of subgroups: "no more than", also add
        to the group. This one done by removing cells until there are mines+1
        left. For example, cell1 cell2, cell3 have 1 mine, which means that
        cell1 and cell2 have no more than 1 mine.
        '''
        for group in self:

            # Same here, no use  doing it to lonely cells out there
            if len(group.cells) == 8:
                continue

            # Here we need >2 cells and >0 mines to create  subgroups
            if len(group.cells) > 2 and group.mines > 0 and \
               group.group_type in ("exactly", "no more than"):

                for cell in group.cells:
                    new_subgroup = MineGroup(group.cells.difference({cell}),
                                             group.mines, "no more than")
                    self.add(new_subgroup)


class MinesweeperSolver:
    ''' Methods related to solving minesweeper game. '''

    def __init__(self, settings=ms.GAME_BEGINNER):
        ''' Initiate the solver. Only requred game settings
        '''

        # Shape, a tuple of dimension sizes
        self.shape = settings.shape
        # N umber of total mines in the game
        self.total_mines = settings.mines

        # Initiate helper (itiration through all cells, neighbours etc)
        self.helper = ms.MinesweeperHelper(self.shape)

        # Placeholder for thej field. Will be populated by self.solve()
        self.field = None

        # Placeholder for all groups. Recalculated for each solver run
        self.groups = AllMineGroups()

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
        # Reset groups and hashes
        self.groups.reset()

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
                    self.groups.add(new_group)

    def method_naive(self):
        ''' Method #1. Naive
        Check if there are all safe or all mines groups.
        Return safe and mines found in those groups
        '''
        safe, mines = [], []
        for group in self.groups:

            # No remaining mines
            if group.is_all_safe():
                safe.extend(list(group.cells))

            # All covered are mines
            if group.is_all_mines():
                mines.extend(list(group.cells))

        return list(set(safe)), list(set(mines))

    @staticmethod
    def deduce_safe(group_a, group_b):
        ''' Given two mine groups, deduce if there are any safe cells.
        '''

        # For that, we need two conditions:
        # 1. A is a subset of B (only checks this way, so extarnal function
        # need to make sure this function called both ways).
        if group_a.cells.issubset(group_b.cells):

            # 2. They have the same number of mines.
            # If so, difference is safe
            # For example, if A(1,2) has one mine and B (1, 2, 3)
            # has 1 mine, cell 3 is safe
            if group_b.mines == group_a.mines:
                return list(group_b.cells - group_a.cells)

        return []

    @staticmethod
    def deduce_mines(group_a, group_b):
        ''' Given two mine groups, deduce if there are any mines.
        '''

        # For that, we need two conditions:
        # 1. A is a subset of B (only checks this way, so extarnal function
        # need to make sure this function called both ways).
        if group_a.cells.issubset(group_b.cells):

            # 2. If difference in number of cells is the same as
            # difference in number of mines: difference is mines
            # For example if A (1, 2) has 1 mine and B (1, 2, 3) has 2 mines,
            # cell 3 is a mine
            if len(group_b.cells - group_a.cells) == \
                    group_b.mines - group_a.mines:
                return list(group_b.cells - group_a.cells)

        return []

    def method_groups(self):
        ''' Method #2. Groups
        Cross check all groups. When group is a subset of
        another group, try to deduce safe or mines.
        '''
        safe, mines = [], []

        # Cross-check all-withall groups
        for group_a in self.groups:
            for group_b in self.groups:

                # Don't compare with itself
                if group_a.hash == group_b.hash:
                    continue

                safe.extend(self.deduce_safe(group_a, group_b))
                mines.extend(self.deduce_mines(group_a, group_b))

                # Difference in cells and mines can also become a new group
                # As lost as one is subset of the other and they have different
                # number of mines
                if group_a.cells.issubset(group_b.cells) and \
                   group_b.mines - group_a.mines > 0:
                    new_group = MineGroup(group_b.cells - group_a.cells,
                                          group_b.mines - group_a.mines)
                    self.groups.add(new_group)

        return list(set(safe)), list(set(mines))

    def method_subgroups(self):
        ''' Subgroups method. Based on breaking groups down "at least" and
        "no more than" subgroups and cross checking them with groups.
        '''
        safe, mines = [], []

        # The idea is similar to the "groups" method:
        # cross-check all the groups, but this time
        # we only will check "at least" and "no more than"
        # subgroups
        for group_a in self.groups:
            for group_b in self.groups:

                # Only compare subgroups "at least" to groups.
                if group_a.group_type == "at least" and \
                   group_b.group_type == "exactly":

                    # Similar to "groups" method: if mines are the same,
                    # the difference is safe
                    # Subgroup A (cells 1, 2) has at least X mines,
                    # Group B (1, 2, 3) has X mines: then cell3 is safe
                    safe.extend(self.deduce_safe(group_a, group_b))

                # Only compare subgroups "no more than" to groups.
                if group_a.group_type == "no more than" and \
                   group_b.group_type == "exactly":

                    # Similar to "groups" method: if mines are the same,
                    # the difference is safe
                    # Subgroup A (cells 1, 2) has at least X mines,
                    # Group B (1, 2, 3) has X mines: then cell3 is safe
                    mines.extend(self.deduce_mines(group_a, group_b))

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

        # 0. First click on the "all 0" corner
        if self.helper.are_all_covered(self.field):
            return [tuple(0 for _ in range(len(self.shape))), ], None

        # Generate groups (main data for basic solving methods)
        self.generate_groups()

        # 1. Naive Method
        #################
        safe, mines = self.method_naive()
        if safe or mines:
            return safe, mines

        # 2. Groups Method
        ##################
        safe, mines = self.method_groups()
        if safe or mines:
            return safe, mines

        # 3. Sub groups Method
        ######################
        # generate subgroups
        # Funny thing, it actually works just as well with only one
        # (either) of these two generated
        self.groups.generate_subgroup_at_least()
        self.groups.generate_subgroup_no_more_than()
        # Actually subgroup method
        safe, mines = self.method_subgroups()
        if safe or mines:
            return safe, mines

        # If there is nothing we can do: return a random cell
        return [self.pick_a_random_cell(covered_cells), ], None


def main():
    ''' Test the solver on a simple game
    '''

    settings = ms.GAME_TEST
    settings = ms.GAME_BEGINNER

    game = ms.MinesweeperGame(settings, seed=1)
    solver = MinesweeperSolver(settings)

    while game.status == ms.STATUS_ALIVE:

        safe, mines = solver.solve(game.uncovered)
        print(f"Move: Safe {safe}, Mines {mines}")
        game.make_a_move(safe, mines)
        print(game)

    print(ms.STATUS_MESSAGES[game.status])


if __name__ == "__main__":
    main()
