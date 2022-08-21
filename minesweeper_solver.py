''' Class for the solver of multidimensional minesweeper game.
Game mechanics and some helper functions are imported from
minesweeper_game.py
'''

import random
import math

import minesweeper_game as ms
import minesweeper_classes as mc


class MinesweeperSolver:
    ''' Methods related to solving minesweeper game. '''

    def __init__(self, settings=ms.GAME_BEGINNER):
        ''' Initiate the solver. Only required game settings
        '''

        # Shape, a tuple of dimension sizes
        self.shape = settings.shape
        # N umber of total mines in the game
        self.total_mines = settings.mines

        # Initiate helper (iteration through all cells, neighbors etc)
        self.helper = ms.MinesweeperHelper(self.shape)

        # Placeholder for the field. Will be populated by self.solve()
        self.field = None

        # Placeholder for the number of remaining mines
        self.remaining_mines = None

        # List of all covered cells
        self.covered_cells = []

        # Placeholder for all groups. Recalculated for each solver run
        self.groups = mc.AllMineGroups()

        # Placeholder for clusters (main element of CSP method)
        self.clusters = []

        # History of clusters we already processed, so we won't have to
        # solved them again (they can be quite time consuming)
        # Also used to cache mine counts, {hash_value: mine_count}
        self.clusters_history = {}

        # Placeholder for mine probability data
        # {cell: probability_of_it_being_a_mine}
        self.probability = None

        # Unaccounted group (all covered minus those  we know the number
        # of mines for). Used to solve exclaves (like 8) and some probability
        # calculations
        self.unaccounted_group = None

        # Info about last move. Normally it would be a tuple
        # ("method name", probability). Probability make sense if it
        # was "Random"
        self.last_move_info = None

    def generate_all_covered(self):
        ''' Return the list of all covered cells
        '''
        all_covered = []
        for cell in self.helper.iterate_over_all_cells():
            if self.field[cell] == ms.CELL_COVERED:
                all_covered.append(cell)
        self.covered_cells = all_covered

    def generate_remaining_mines(self):
        ''' Based on mines we know about, calculate how many are still
        on the filed. Populate self.remaining_mines
        '''
        self.remaining_mines = self.total_mines
        for cell in self.helper.iterate_over_all_cells():
            if self.field[cell] == ms.CELL_MINE:
                self.remaining_mines -= 1

    def generate_groups(self):
        '''Populate self.group with all found cell groups
        '''
        # Reset groups and hashes
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
                if self.field[neighbor] == ms.CELL_COVERED:
                    covered_neighbors.append(neighbor)
                # Subtract all marked mines
                if self.field[neighbor] == ms.CELL_MINE:
                    active_mines -= 1

            # If the list of covered cells is not empty:
            # store it in the self.groups
            if covered_neighbors:
                new_group = mc.MineGroup(covered_neighbors, active_mines)
                self.groups.add(new_group)

    def generate_clusters(self):
        '''Populate self.clusters with cell clusters
        '''
        # Reset clusters
        self.clusters = []
        # Reset all "belong to cluster" information from the groups
        self.groups.reset_clusters()

        # Keep going as long as there are groups not belonging to clusters
        while self.groups.next_non_clustered_groups() is not None:

            # Initiate a new cluster with such a group
            new_cluster = mc.CellCluster(
                self.groups.next_non_clustered_groups())

            while True:

                # Look through all groups
                for group in self.groups:
                    # If it overlaps with this group and not part of any
                    # other cluster - add this group
                    if group.group_type == "exactly" and \
                       group.belong_to_cluster is None and \
                       new_cluster.overlap(group):
                        new_cluster.add(group)
                        break

                # We went through the groups without adding any:
                # new_cluster is done
                else:
                    self.clusters.append(new_cluster)
                    # Exit the while loop
                    break

    def generate_unaccounted(self):
        ''' Calculate the cells and mines of "unknown" area, that is covered
        area minus what we know from groups. Used in method "Coverage" and
        allows for better background probability calculations.
        '''
        # Reset
        self.unaccounted_group = None

        # Initiate by having a mutable copy of all cells and all mines
        accounted_cells = set()
        accounted_mines = 0

        while True:
            # The idea is to find a group that has a largest number of
            # unaccounted cells
            best_count = None
            best_group = None
            for group in self.groups:

                # We only need "exactly" groups
                if group.group_type != "exactly":
                    continue

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

        # unaccounted cells are all minus accounted
        unaccounted_cells = set(self.covered_cells).difference(accounted_cells)
        # Same with mines
        unaccounted_mines = self.remaining_mines - accounted_mines

        # Those unaccounted mines can now for a new  group
        self.unaccounted_group = mc.MineGroup(unaccounted_cells,
                                              unaccounted_mines)

    @staticmethod
    def deduce_safe(group_a, group_b):
        ''' Given two mine groups, deduce if there are any safe cells.
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
        Similar to deduce_safe, but the condition is different.
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

    def method_groups(self):
        ''' Method #2. Groups
        Cross check all groups. When group is a subset of
        another group, try to deduce safe or mines.
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
                    self.groups.add(new_group)

        return list(set(safe)), list(set(mines))

    def method_subgroups(self):
        ''' Subgroups method. Based on breaking groups down "at least" and
        "no more than" subgroups and cross checking them with groups.
        '''
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

    def method_csp(self):
        ''' CSP method. We look at the overlapping groups to check if some
        cells are always safe or mines in all valid solutions.
        '''
        safe, mines = [], []

        # Generate clusters
        self.generate_clusters()
        for cluster in self.clusters:

            # Cluster deduplication. If we solved this cluster already, ignore
            # it, but get the probable mines  from the "cache" - we'll need it
            # for probability method
            if cluster.calculate_hash() in self.clusters_history:
                cluster.probable_mines = \
                    self.clusters_history[cluster.calculate_hash()]
                continue

            # Solve the cluster, including weights,
            # frequencies and probable mines
            cluster.solve_cluster(self.remaining_mines)
            cluster.calculate_solution_weights(self.covered_cells,
                                               self.remaining_mines)
            cluster.calculate_frequencies()
            cluster.possible_mine_counts()

            # Get safe cells and mines from cluster
            safe.extend(cluster.safe_cells())
            mines.extend(cluster.mine_cells())

            # Save cluster's hash in history for deduplication
            self.clusters_history[cluster.calculate_hash()] = \
                cluster.probable_mines

        return list(set(safe)), list(set(mines))

    def method_coverage(self):
        '''Extract safes and  mines from the unaccounted group
        '''
        if self.unaccounted_group is None:
            return [], []

        safe, mines = [], []
        if self.unaccounted_group.is_all_safe():
            safe.extend(list(self.unaccounted_group.cells))
        if self.unaccounted_group.is_all_mines():
            mines.extend(list(self.unaccounted_group.cells))
        return list(set(safe)), list(set(mines))

    def calculate_probabilities(self):
        ''' Calculate probabilities of mines (and populate
        self.mine_probabilities, using several methods
        '''

        def background_probabilities(self):
            ''' Background probability is all mines / all covered cells .
            It is quite crude and often inaccurate, but sometimes it is better
            to click deep into the unknown rather than try 50/50 guess.
            '''
            background_probability = \
                self.remaining_mines / len(self.covered_cells)
            for cell in self.covered_cells:
                self.probability[cell] = \
                    mc.CellProbabilityInfo(background_probability,
                                           "Background")

        def probabilities_for_groups(self):
            ''' Populate self.mine_probabilities, based on groups data
            Each cell's probability is: max(group_mines / group_cells, ...),
            for all cells it is in.
            '''
            for group in self.groups:
                # We only need "exactly" type of groups
                if group.group_type != "exactly":
                    continue

                # Probability of each mine in teh group
                group_probability = group.mines / len(group.cells)
                for cell in group.cells:
                    # If group's probability is higher than the background:
                    # Overwrite the probability result
                    if group_probability > \
                       self.probability[cell].mine_chance:
                        self.probability[cell] = \
                            mc.CellProbabilityInfo(group_probability, "Groups")

        def csp_probabilities(self):
            ''' Calculate mine possibilities based on CSP
            clusters
            '''
            for cluster in self.clusters:
                for cell, frequency in cluster.frequencies.items():
                    # Overwrite the probability result
                    self.probability[cell] = \
                        mc.CellProbabilityInfo(frequency, "CSP")

        def cluster_leftovers_probabilities(self):
            ''' Estimate the mine numbers in clusters, and deduce the chance
            of mines outside of clusters. Basically, it is a more accurate
            version of background probabilities.
            '''
            # Rarely, there are no clusters (when covered area is walled off
            # by mines, for example, like last 8)
            if not self.clusters:
                return

            # Collect all cells and mine estimations from all clusters
            cells_in_clusters = set()
            cross_mines = None

            # First, collect all cells that don't belong to any cluster
            for cluster in self.clusters:
                cells_in_clusters = cells_in_clusters.union(cluster.cells_set)
            leftover_cells = set(self.covered_cells).\
                difference(cells_in_clusters)

            # If no unaccounted cells - can't do anything
            if not leftover_cells:
                return

            # Next, calculate accounted mines and solutions for all cluster's
            # cross-matching. That is, all mines in all clusters: all
            # solutions where we have that many mines
            for cluster in self.clusters:

                # Calculate all possible mine counts in clusters and the number
                # of solutions permutations foreach count
                if cross_mines is None:
                    cross_mines = cluster.probable_mines.copy()
                else:
                    new_cross_mines = {}
                    # These two loops allows to cross-check all clusters and
                    # have a list of all permutations (where mines are added
                    # and mine counts multiplied)
                    for already_mines, already_count in cross_mines.items():
                        for mines, count in cluster.probable_mines.items():
                            new_cross_mines[already_mines + mines] = \
                                already_count * count
                    cross_mines = new_cross_mines

            # Weight of each mine count: number of possible combinations
            # in the leftover cells.
            # Last line here is case there are permutations that exceed
            # the total number of remaining mines
            cross_mines_weights = {mines: math.comb(len(leftover_cells),
                                   self.remaining_mines - mines) * solutions
                                   for mines, solutions in cross_mines.items()
                                   if self.remaining_mines - mines >= 0}

            # Total weights for all mine counts
            total_weights = sum(cross_mines_weights.values())

            # If the cluster was not solved total weight would be zero.
            if total_weights == 0:
                return

            # Total number of mines in all clusters
            # (sum of count * probability)
            mines_in_clusters = sum(mines * weight / total_weights
                                    for mines, weight
                                    in cross_mines_weights.items())

            # And this is the probability of a mine in those cells
            leftover_mine_chance = (self.remaining_mines - mines_in_clusters) \
                / len(leftover_cells)

            # Fill in the probabilities
            for cell in leftover_cells:
                self.probability[cell] = \
                    mc.CellProbabilityInfo(leftover_mine_chance,
                                           "CSP Leftovers")

        # Reset probabilities
        self.probability = mc.AllCellsProbability()
        # Background probability: all remaining mines on all covered cells
        background_probabilities(self)
        # Based on mines in groups
        probabilities_for_groups(self)
        # Based on CSP solutions
        csp_probabilities(self)
        # Probabilities of non-cluster mines
        cluster_leftovers_probabilities(self)

    def calculate_opening_chances(self):
        ''' Populate opening_chance in self.probabilities by looking
        at neighbors' chances
        '''
        # Go through all cells we have probability info for
        # (that would be all covered cells)
        for cell, cell_info in self.probability.items():
            zero_chance = 1
            # Look at neighbors of each cell
            for neighbor in self.helper.cell_surroundings(cell):
                # If there are any mines around, there is no chance of opening
                if self.field[neighbor] == ms.CELL_MINE:
                    cell_info.opening_chance = 0
                    break
                # Otherwise each mine chance decrease opening chance
                # by (1 - mine chance) times
                if neighbor in self.probability:
                    zero_chance *= (1 - self.probability[neighbor].mine_chance)
            else:
                self.probability[cell].opening_chance = zero_chance

    @staticmethod
    def pick_a_random_cell(cells):
        '''Pick a random cell out of the list of cells.
        (Either for testing or when we are reduced to guessing)
        '''
        return random.choice(cells)

    def solve(self, field, use_probability=True):
        ''' Main solving function.
        Go through various solving methods and return safe and mines lists
        as long as any of the methods return results
        In:
        - the field (what has been uncovered so far).
        - use_probability: True for regular use. False for using only
        deterministic methods: this  mode is to estimate worth of next moves
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
        self.generate_remaining_mines()
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
                ):
            safe, mines = method()
            if safe or mines:
                self.last_move_info = (method_name, None, None)
                return safe, mines

        # Deterministic methods are over. If we  are not supposed to use
        # probability-based ones, our job here is done.
        if not use_probability:
            return safe, mines

        # Calculate mine probability using various methods
        self.calculate_probabilities()
        # Calculate opening chances
        self.calculate_opening_chances()

        # Pick a cell that is least likely a mine
        lucky_cells = self.probability.pick_lowest_probability(self)

        if lucky_cells:
            lucky_cell = self.pick_a_random_cell(lucky_cells)
            self.last_move_info = ("Probability",
                                   self.probability[lucky_cell].source,
                                   self.probability[lucky_cell].mine_chance)
            return [lucky_cell, ], None

        # Until the count method implemented, there is a chance of
        # "exclave" cells. This is a catch-all for that
        self.last_move_info = ("Last Resort", None)
        return [self.pick_a_random_cell(self.covered_cells), ], None


def main():
    ''' Test the solver on a simple game
    '''

    settings = ms.GAME_TEST
    settings = ms.GAME_BEGINNER

    game = ms.MinesweeperGame(settings, seed=0)
    solver = MinesweeperSolver(settings)

    while game.status == ms.STATUS_ALIVE:

        safe, mines = solver.solve(game.uncovered)
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

    print(ms.STATUS_MESSAGES[game.status])


if __name__ == "__main__":
    main()
