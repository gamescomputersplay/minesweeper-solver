''' Solver class for (multidimensional) minesweeper game.
It would receive a current field and return cells that are
[most probably] safe or contain mines.
'''

import random
import math

import minesweeper_game as mg
import minesweeper_classes as mc


class MinesweeperSolver:
    ''' Main solver class. Hold information about current game state
    (field, mines), some calculated data (mine groups, clusters,
    probability data). Methods to calculate safe and mines
    from the current game state.
    '''

    def __init__(self, settings=mg.GAME_BEGINNER, helper=None):
        ''' Initiate the solver.
        IN:
        - settings: GameSettings object with field size and total mines
        - helper: a MinesweeperHelper object with some precalculated things.
          it can be passed in to save time and not having to recalculate it.
        '''

        # Shape, a tuple of dimensions of the game field
        self.shape = settings.shape
        # Number of total initial mines
        self.total_mines = settings.mines

        # initiate helper (iteration through all cells, neighbors etc)
        # Or use one that was passed in.
        if helper is None:
            self.helper = mg.MinesweeperHelper(self.shape)
        else:
            self.helper = helper

        # Placeholder for the field. Will be populated by self.solve()
        self.field = None

        # Placeholder for the number of remaining mines
        # will be calculated by "calculate_remaining mines"
        self.remaining_mines = None

        # List of all covered cells
        # Populated by "generate_all_covered"
        self.covered_cells = []

        # Placeholder for all mine groups.
        # Populated by "generate_groups"
        self.groups = mc.AllGroups()

        # Placeholder for clusters (collections of groups)
        # Populated by method_csp
        self.all_clusters = mc.AllClusters(self.covered_cells,
                                           self.remaining_mines, self.helper)

        # Placeholder for mine probability data
        # {cell: CellProbability object}
        # Calculated by "calculate_probabilities"
        self.probability = None

        # Unaccounted group (one that is away from all the "numbers").
        # Used to solve exclaves (like 8) and some probability calculations
        # Populated by "generate_unaccounted"
        self.unaccounted_group = None

        # Placeholder for bruteforce solutions (works only when few enough
        # cells are left). Used by both deterministic and probability based
        # methods. Populated by generate_bruteforce
        self.bruteforce_solutions = None

        # Info about last move. Will be used by simulator to analyze
        # performance of different methodsNormally.
        # Is a tuple with 3 elements
        # ("Method name", "Probability sub method name", Probability of a mine)
        self.last_move_info = None

    def copy(self):
        ''' Create a copy of solver object.
        Reuse settings and helper from the original one.
        Copies are used to look into the 2nd move.
        '''
        settings = mg.GameSettings(self.shape, self.total_mines)
        new_solver = MinesweeperSolver(settings, helper=self.helper)
        return new_solver

    def generate_all_covered(self):
        ''' Populate self.covered_cells with the list of all covered cells.
        '''
        all_covered = []
        for cell in self.helper.iterate_over_all_cells():
            if self.field[cell] == mg.CELL_COVERED:
                all_covered.append(cell)
        self.covered_cells = all_covered

    def calculate_remaining_mines(self):
        ''' Populate self.remaining_mines with the number of mines
        that are still not flagged.
        '''
        self.remaining_mines = self.total_mines
        for cell in self.helper.iterate_over_all_cells():
            if self.field[cell] == mg.CELL_MINE:
                self.remaining_mines -= 1

    def generate_groups(self):
        ''' Populate self.group with MineGroup objects
        '''

        # Reset the groups
        self.groups.reset()

        # Go over all cells and find all the "Numbered ones"
        for cell in self.helper.iterate_over_all_cells():

            # Groups are only for numbered cells
            if self.field[cell] <= 0:
                continue

            # For them we'll need to know two things:
            # What are the uncovered cells around it
            covered_neighbors = []
            # And how many "Active" (that is, minus marked)
            # mines are still there
            active_mines = self.field[cell]

            # Go through the neighbors
            for neighbor in self.helper.cell_surroundings(cell):
                # Collect all covered cells
                if self.field[neighbor] == mg.CELL_COVERED:
                    covered_neighbors.append(neighbor)
                # Subtract all marked mines
                if self.field[neighbor] == mg.CELL_MINE:
                    active_mines -= 1

            # If the list of covered cells is not empty:
            # store it in the self.groups
            if covered_neighbors:
                new_group = mc.MineGroup(covered_neighbors, active_mines)
                self.groups.add_group(new_group)

    def generate_clusters(self):
        ''' Initiate self.all_clusters and populate it with
        GroupCluster objects
        '''
        # Reset clusters
        self.all_clusters = mc.AllClusters(self.covered_cells,
                                           self.remaining_mines, self.helper)
        # Reset all "belong to cluster" information from the groups
        self.groups.reset_clusters()

        # Keep going as long as there are groups not belonging to clusters
        while self.groups.next_non_clustered_groups() is not None:

            # Initiate a new cluster with such a group
            new_cluster = mc.GroupCluster(
                self.groups.next_non_clustered_groups())

            while True:

                # Look through all groups
                for group in self.groups.exact_groups():
                    # If it overlaps with this group and not part of any
                    # other cluster - add this group
                    if group.belong_to_cluster is None and \
                       new_cluster.overlap(group):
                        new_cluster.add_group(group)
                        break

                # We went through the groups without adding any:
                # new_cluster is done
                else:
                    self.all_clusters.clusters.append(new_cluster)
                    # Exit the while loop
                    break

    def generate_unaccounted(self):
        ''' Populate self.unaccounted_group with a MineGroup made of cells
        and mines from "unknown" area, that is NOT next to any number.
        Used in "Coverage" method and in mine probability calculations.
        '''

        def coverage_attempt(accounted_cells, accounted_mines):
            ''' Create a coverage (set of non-overlapping mine groups),
            given cells and mines that are already in the coverage.
            Uses greedy method to maximize the number of cells in the coverage
            '''

            while True:
                # The idea is to find a group that has a largest number of
                # unaccounted cells
                best_count = None
                best_group = None
                for group in self.groups.exact_groups():

                    # If group overlaps with what we have so far -
                    # we don't need such group
                    if accounted_cells.intersection(group.cells):
                        continue

                    # Find the biggest group that we haven't touched yet
                    if best_count is None or len(group.cells) > best_count:
                        best_count = len(group.cells)
                        best_group = group

                # We have a matching group
                if best_group is not None:
                    # Cells from that group from now on are accounted for
                    accounted_cells = accounted_cells.union(best_group.cells)
                    # And so are  mines
                    accounted_mines += best_group.mines
                # No such  group was found: coverage is done
                else:
                    break

            return accounted_cells, accounted_mines

        # Reset the final variable
        self.unaccounted_group = None

        # This method usually has no effect in the beginning of the game.
        # Highest count of remaining cells when it worked was at 36-37.
        if len(self.covered_cells) > 40:
            return

        # Generate several coverage options.
        # Put them into coverage_options
        coverage_options = []
        # FOr each option we are going to start with different group,
        # and then proceed with greedy algorithm
        for group in self.groups:
            initial_cells = set(group.cells)
            initial_mines = group.mines
            coverage_option_cells, coverage_option_mines = \
                coverage_attempt(initial_cells, initial_mines)
            coverage_options.append((coverage_option_cells,
                                     coverage_option_mines))

        if not coverage_options:
            return

        # Sort them by the number of cells in coverage
        # Choose the one with the most cells
        coverage_options.sort(key=lambda x: len(x[0]), reverse=True)
        accounted_cells, accounted_mines = coverage_options[0]

        # unaccounted cells are all minus accounted
        unaccounted_cells = set(self.covered_cells).difference(accounted_cells)
        # Same with mines
        unaccounted_mines = self.remaining_mines - accounted_mines

        # Those unaccounted mines can now for a new  group
        self.unaccounted_group = mc.MineGroup(unaccounted_cells,
                                              unaccounted_mines)

    def generate_bruteforce(self):
        ''' Generate and populate bruteforce_solutions. Solutions are list of
        lists [True, False ..], where True stands for a mine and position is
        the position of cells in self.cells.
        '''
        #for group in self.groups.exact_groups():
        #    print (group)
        # dict to simplify looking for cells in solutions
        cells_positions = {cell: pos for pos, cell in enumerate(self.covered_cells)}
        #print(cells_positions)

        # Generate all possible combinations of mines
        permutations = mc.all_mines_positions(len(self.covered_cells), self.remaining_mines)
        #print(len(permutations))

        # Filter, keeping only those that comply with all groups
        filtered_permutations = []
        for permutation in permutations:
            for group in self.groups.exact_groups():
                # Count mines in the solution in cells of teh group
                mine_count = 0
                for cell in group.cells:
                    if permutation[cells_positions[cell]]:
                        mine_count += 1
                # If count doesn't match - next solution
                if mine_count != group.mines:
                    break
            # If all groups were satisfied - copy solution to filtered solutions
            else:
                filtered_permutations.append(permutation)

        self.bruteforce_solutions = filtered_permutations

    @staticmethod
    def deduce_safe(group_a, group_b):
        ''' Given two mine groups, deduce if there are any safe cells.
        if found, return list of safe cells.
        '''
        # For that, we need two conditions:
        # 1. A is a subset of B (only checks this way, so external function
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
        Similar to deduce_safe, but for mines.
        '''
        if group_a.cells.issubset(group_b.cells):

            # 2. If difference in number of cells is the same as
            # difference in number of mines: difference is mines
            # For example if A (1, 2) has 1 mine and B (1, 2, 3) has 2 mines,
            # cell 3 is a mine
            if len(group_b.cells - group_a.cells) == \
                    group_b.mines - group_a.mines:
                return list(group_b.cells - group_a.cells)

        return []

    def method_naive(self):
        ''' Method #1. Naive.
        Try to find safe and mines in the groups themselves.
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

    def method_groups(self):
        ''' Method #2. Groups.
        Cross check all groups. When group is a subset of
        another group, try to deduce safe cells and mines.
        '''
        safe, mines = [], []

        # Cross-check all-with-all groups
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
                # len(group_b.cells) < 8 prevents computational explosion on
                # multidimensional fields
                if len(group_b.cells) < 8 and \
                   group_a.cells.issubset(group_b.cells) and \
                   group_b.mines - group_a.mines > 0:
                    new_group = mc.MineGroup(group_b.cells - group_a.cells,
                                             group_b.mines - group_a.mines)
                    self.groups.add_group(new_group)

        return list(set(safe)), list(set(mines))

    def method_subgroups(self):
        ''' Method #3. Subgroups. Breaking down groups into "subgroups":
        "at least" and "no more than". Cross-check them with regular groups
        to deduce mines.
        '''
        # Note how many groups we have
        self.groups.count_groups = len(self.groups.mine_groups)

        # Generate subgroups
        # Funny thing, it actually works just as well with only one
        # (either) of these two generated
        self.groups.generate_subgroup_at_least()
        self.groups.generate_subgroup_no_more_than()

        safe, mines = [], []

        # The idea is similar to the "groups" method:
        # cross-check all the groups, but this time
        # we only will check "at least" and "no more than"
        # subgroups
        # Group A are all subgroups (at least, no more)
        for group_a in self.groups.subgroups():
            # Group B are all groups (exactly)
            for group_b in self.groups.exact_groups():

                # Only compare subgroups "at least" to groups.
                if group_a.group_type == "at least":

                    # Similar to "groups" method: if mines are the same,
                    # the difference is safe
                    # Subgroup A (cells 1, 2) has at least X mines,
                    # Group B (1, 2, 3) has X mines: then cell3 is safe
                    safe.extend(self.deduce_safe(group_a, group_b))

                # Only compare subgroups "no more than" to groups.
                if group_a.group_type == "no more than":

                    # Similar to "groups" method: if mines are the same,
                    # the difference is safe
                    # Subgroup A (cells 1, 2) has at least X mines,
                    # Group B (1, 2, 3) has X mines: then cell3 is safe
                    mines.extend(self.deduce_mines(group_a, group_b))

        return list(set(safe)), list(set(mines))

    def method_csp(self):
        ''' Method #4. CSP (Constraint Satisfaction Problem).
        Generate overlapping groups (clusters). For each cluster find safe
        cells and mines by brute forcing all possible solutions.
        '''
        safe, mines = [], []

        # Generate clusters
        self.generate_clusters()
        # DO all teh solving / calculate frequencies stuff
        self.all_clusters.calculate_all()

        for cluster in self.all_clusters.clusters:

            # Get safe cells and mines from cluster
            safe.extend(cluster.safe_cells())
            mines.extend(cluster.mine_cells())

        return list(set(safe)), list(set(mines))

    def method_coverage(self):
        ''' Method #5, Coverage.
        Deduce safes and  mines from the "unaccounted" group
        '''
        # Trivial coverage cases: no mines and all mines
        if self.remaining_mines == 0:
            return self.covered_cells, []
        if len(self.covered_cells) == self.remaining_mines:
            return [], self.covered_cells

        if self.unaccounted_group is None:
            return [], []

        safe, mines = [], []
        if self.unaccounted_group.is_all_safe():
            safe.extend(list(self.unaccounted_group.cells))
        if self.unaccounted_group.is_all_mines():
            mines.extend(list(self.unaccounted_group.cells))
        return list(set(safe)), list(set(mines))

    def method_bruteforce(self):
        '''Bruteforce mine probabilities, when there are only a handful
        of cells/mines left. This replaces more sophisticate
        calculate_probabilities method.
        '''
        # Use this method only if there is not a lot combinations to go through
        if len(self.covered_cells) > 25 or \
           math.comb(len(self.covered_cells), self.remaining_mines) > 3060:
            return [], []

        safe, mines = [], []
        self.generate_bruteforce()

        # Go through all cells
        for position, cell in enumerate(self.covered_cells):
            # And count mines in all solutions in this position
            solutions_with_mines = 0
            for solution in self.bruteforce_solutions:
                if solution[position]:
                    solutions_with_mines += 1

            # If there were no mines - this cell is safe
            if solutions_with_mines == 0:
                safe.append(cell)
            # If there were as many mines as solutions - it's a mine
            elif solutions_with_mines == len(self.bruteforce_solutions):
                mines.append(cell)

        return safe, mines

    def bruteforce_probabilities(self):
        '''Bruteforce mine probabilities, when there are only a handful
        of cells/mines left. This replaces more sophisticate
        calculate_probabilities method.
        '''

    def calculate_probabilities(self):
        ''' Final method. "Probability". Use various methods to determine
        which cell(s) is least likely to have a mine
        '''

        def background_probabilities(self):
            ''' Populate self.probabilities based on background probability.
            Which is All mines divided by all covered cells.
            It is quite crude and often inaccurate, it is just a fallback
            if any of more accurate methods don't work.
            '''
            background_probability = \
                self.remaining_mines / len(self.covered_cells)

            for cell in self.covered_cells:
                self.probability.cells[cell] = \
                    mc.CellProbability(cell, "Background",
                                       background_probability)

        def probabilities_for_groups(self):
            ''' Update self.probabilities, based on mine groups.
            For each group consider mine probability as "number of mines
            divided by the number of cells".
            '''
            for group in self.groups.exact_groups():

                # Probability of each mine in teh group
                group_probability = group.mines / len(group.cells)
                for cell in group.cells:

                    # If group's probability is higher than the background:
                    # Overwrite the probability result
                    if group_probability > \
                       self.probability.cells[cell].mine_chance:
                        self.probability.cells[cell] = \
                            mc.CellProbability(cell, "Groups", group_probability)

        def csp_probabilities(self):
            ''' Update self.probabilities based on results from CSP method.
            '''
            for cluster in self.all_clusters.clusters:
                for cell, frequency in cluster.frequencies.items():
                    # Overwrite the probability result
                    self.probability.cells[cell] = \
                        mc.CellProbability(cell, "CSP", frequency)

        def cluster_leftovers_probabilities(self):
            ''' Update self.probabilities based on "leftovers",
            cells that are not in any clusters. (not to be confused with
            "Unaccounted" - those are cells that are not in any group).
            '''
            self.all_clusters.calculate_leftovers()

            # For some reasons, calculation failed
            # (probably clusters were too long, or unsolvable)
            if self.all_clusters.leftover_mine_chance is None:
                return

            # Fill in the probabilities
            for cell in self.all_clusters.leftover_cells:
                self.probability.cells[cell] = \
                    mc.CellProbability(cell, "CSP Leftovers",
                                       self.all_clusters.leftover_mine_chance)

        # Reset probabilities
        self.probability = mc.AllProbabilities()
        # Background probability: all remaining mines on all covered cells
        background_probabilities(self)
        # Based on mines in groups
        probabilities_for_groups(self)
        # Based on CSP solutions
        csp_probabilities(self)
        # Probabilities of non-cluster mines
        cluster_leftovers_probabilities(self)

    def calculate_opening_chances(self):
        ''' Populate opening_chance in self.probabilities: a chance that this
        cell is a zero. (Which is a good thing)
        '''
        # Go through all cells we have probability info for
        # (that would be all covered cells)
        for cell, cell_info in self.probability.cells.items():
            zero_chance = 1
            # Look at neighbors of each cell
            for neighbor in self.helper.cell_surroundings(cell):
                # If there are any mines around, there is no chance of opening
                if self.field[neighbor] == mg.CELL_MINE:
                    cell_info.opening_chance = 0
                    break
                # Otherwise each mine chance decrease opening chance
                # by (1 - mine chance) times
                if neighbor in self.probability.cells:
                    zero_chance *= (1 - self.probability.cells[neighbor].mine_chance)
            else:
                self.probability.cells[cell].opening_chance = zero_chance

    def calculate_frontier(self):
        ''' Populate frontier (how many groups may be affected by this cell)
        '''
        # Generate frontier
        self.groups.generate_frontier()

        for cell in self.groups.frontier:
            for neighbors in self.helper.cell_surroundings(cell):
                if neighbors in self.probability.cells:
                    self.probability.cells[neighbors].frontier += 1

    def calculate_next_safe_csp(self):
        ''' Populate "next safe" information (how many guaranteed safe cells
        will be in the next move, based on CSP solutions).
        '''
        # Do the calculations
        self.all_clusters.calculate_all_next_safe()

        # Populate probability object with this info
        for cluster in self.all_clusters.clusters:
            for cell, next_safe in cluster.next_safe.items():
                self.probability.cells[cell].csp_next_safe = next_safe

    @staticmethod
    def pick_a_random_cell(cells):
        '''Pick a random cell out of the list of cells.
        (Either for testing or when we are reduced to guessing from list
        of cells with exactly the same probabilities)
        '''
        return random.choice(cells)

    def solve(self, field, next_moves=1):
        ''' Main solving function.
        Go through various solving methods and return safe and mines lists
        as long as any of the methods return results
        In:
        - the field (what has been uncovered so far).
        - next_moves: remaining levels of recursion. 0 - don't look into
          future boards. 1 - look 1 move ahead etc
        Out:
        - list of safe cells
        - list of mines
        '''
        # Store field as an instance variable
        self.field = field

        # First click on the "all 0" corner
        if self.helper.are_all_covered(self.field):
            self.last_move_info = ("First click", None, None)
            all_zeros = tuple(0 for _ in range(len(self.shape)))
            return [all_zeros, ], None

        # Several calculation needed for the following solution methods
        # A list of all covered cells
        self.generate_all_covered()
        # Number of remaining mines
        self.calculate_remaining_mines()
        # Generate groups (main data for basic solving methods)
        self.generate_groups()
        # Unaccounted cells (covered minus mines, has  to go after the  groups)
        self.generate_unaccounted()

        # Try 4 deterministic methods
        # If any yielded result - return it
        for method, method_name in (
                (self.method_naive, "Naive"),
                (self.method_groups, "Groups"),
                (self.method_subgroups, "Subgroups"),
                (self.method_csp, "CSP"),
                (self.method_coverage, "Coverage"),
                #(self.method_bruteforce, "Bruteforce"),
                ):
            safe, mines = method()
            if safe or mines:
                self.last_move_info = (method_name, None, None)
                return safe, mines

        # Bruteforce probabilities will go here
        if len(self.covered_cells) <= -1: #16:
            # Bruteforce-based calculation of probabilities
            # if there are not a lot cells and mines left
            self.bruteforce_probabilities()
        # Otherwise use more complex approach
        else:
            # Calculate mine probability using various methods
            self.calculate_probabilities()
            # Calculate safe cells for teh next move in CSP
            self.calculate_next_safe_csp()

        # Two more calculations that will be used to pick
        # the best random cell:
        # Opening chances (chance that cell is a zero)
        self.calculate_opening_chances()
        # Does it touch a frontier (cells that already are in groups)
        self.calculate_frontier()

        # Get cells that is least likely a mine
        lucky_cells = \
            self.probability.get_luckiest(self.all_clusters, next_moves, self)

        if lucky_cells:
            # There may be more than one such cells, pick a random one
            lucky_cell = self.pick_a_random_cell(lucky_cells)
            # Store information about expected chance of mine and how
            # this chance was calculated
            self.last_move_info = ("Probability",
                                   self.probability.cells[lucky_cell].source,
                                   self.probability.cells[lucky_cell].mine_chance)
            return [lucky_cell, ], None

        # This should not happen, but here's a catch-all if it does
        self.last_move_info = ("Last Resort", None)
        return [self.pick_a_random_cell(self.covered_cells), ], None


def main():
    ''' Test the solver on one game.
    '''

    settings = mg.GAME_TEST
    settings = mg.GAME_BEGINNER
    settings = mg.GAME_INTERMEDIATE
    # settings = mg.GAME_EXPERT

    game = mg.MinesweeperGame(settings, seed=2)
    solver = MinesweeperSolver(settings)

    while game.status == mg.STATUS_ALIVE:

        safe, mines = solver.solve(game.uncovered, next_moves=0)
        method, random_method, chance = solver.last_move_info

        chance_str, random_method_str = "", ""
        if chance is not None:
            chance_str = f"Mine chance: {chance}, "

        if random_method is not None:
            random_method_str = f" ({random_method})"

        print(f"Method: {method}{random_method_str}, {chance_str}" +
              f"Safe: {safe}, Mines: {mines}")

        game.make_a_move(safe, mines)
        print(game)

    print(mg.STATUS_MESSAGES[game.status])


if __name__ == "__main__":
    main()
