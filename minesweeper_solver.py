''' Class for the solver of multidimensional minesweeper game.
Game mechanics and some helper functions are imported from
minesweeper_game.py
'''

import random
import math
import minesweeper_game as ms


class MineGroup:
    ''' A MineGroup is a set of cells that are known
    to have a certain number of mines.
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

        # Group type ("exactly", "no more than", "at least")
        self.group_type = group_type

        # Calculate hash (for deduplication)
        self.hash = self.calculate_hash()

        # Placeholder for cluster information
        # (each group can belong to only one cluster)
        self.belongs_to_cluster = None

    def is_all_safe(self):
        ''' If group has 0 mines, it is safe
        '''
        return self.mines == 0

    def is_all_mines(self):
        ''' If group has as many cells as mines - they are all mines
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
        out = f"Cell(s) ({len(self.cells)}) "
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
        # List of MineGroups
        self.mine_groups = []

    def reset(self):
        ''' Clear the data
        '''
        self.hashes = set()
        self.mine_groups = []

    def reset_clusters(self):
        ''' Clear "belong to cluster" for each group.
        '''
        for group in self.mine_groups:
            group.belong_to_cluster = None

    def next_non_clustered_groups(self):
        ''' Return first found group, that is not a part of a cluster
        '''
        for group in self.mine_groups:
            if group.belong_to_cluster is None and \
               group.group_type == "exactly":
                return group
        return None

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
            if len(group.cells) > 7:
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
            if len(group.cells) > 7:
                continue

            # Here we need >2 cells and >0 mines to create  subgroups
            if len(group.cells) > 2 and group.mines > 0 and \
               group.group_type in ("exactly", "no more than"):

                for cell in group.cells:
                    new_subgroup = MineGroup(group.cells.difference({cell}),
                                             group.mines, "no more than")
                    self.add(new_subgroup)

    def __str__(self):
        ''' Some info about the groups (for debugging)
        '''
        return f"MineGroups contains {len(self.mine_groups)} groups"


class CellCluster:
    ''' CellCluster is a group of cells connected together by an overlapping
    list of groups. In other words mine/safe in any of the cell, can
    potentially trigger safe/mine in any other cell of the cluster.
    Is a basic class for CSP (constraint satisfaction problem) method
    '''

    def __init__(self, group=None):
        # List of cells in the cluster
        self.cells_set = set()
        # List if groups in the cluster
        self.groups = []

        # Initiate by adding the first group
        if group is not None:
            self.add(group)

        # Placeholder for a list of set (we need them in a fixed order)
        self.cells = []

        # Placeholder for the solutions of this CSP
        # (all valid sets of mines and safe cells)
        # Positions corresponds to self.cells
        self.solutions = []

        # Placeholder for the resulting frequencies of mines in each cell
        self.frequencies = {}

        # Placeholder for solution weight - how probable is this solution
        # based on the number of mines in it
        self.solution_weights = []

        # Estimated number of mines in cluster (adjusted for the different
        # weight of different solutions)
        self.probable_mines = None

    def add(self, group):
        ''' Adding group to a cluster (assume they overlap).
        Add group's cells to the set of cells
        Also, mark the group as belonging to this cluster
        '''
        # Total list of cells in the cluster (union of all groups)
        self.cells_set = self.cells_set.union(group.cells)
        # List of groups belonging to the cluster
        self.groups.append(group)
        # Mark the group that it has been used
        group.belong_to_cluster = self

    def overlap(self, group):
        ''' Check if cells in group overlap with cells in the cluster.
        '''
        return len(self.cells_set & group.cells) > 0

    @staticmethod
    def all_mines_positions(cells_count, mines_to_set):
        ''' Generate all permutations for "Choose k from n",
        which is equivalent to all possible ways mines are
        located in the cells.
        Result is a list of tuples like (False, False, True, False),
        indicating if the item was chosen (if there is a mine in the cell).
        For example, for generate_mines_permutations(2, 1) the output is:
        [(True, False), (False, True)]
        '''

        def recursive_choose_generator(current_combination, mines_to_set):
            ''' Recursive part of "Choose without replacement" permutation
            generator, results are put into outside "result" variable
            '''
            # No mines to set: save results, out of the recursion
            if mines_to_set == 0:
                result.add(tuple(current_combination))
                return

            for position, item in enumerate(current_combination):
                # Find all the "False" (not a mine) cells
                if not item:
                    # Put a mine in it, go to the next recursion level
                    current_copy = current_combination.copy()
                    current_copy[position] = True
                    recursive_choose_generator(current_copy, mines_to_set - 1)

        result = set()
        all_cells_false = [False for _ in range(cells_count)]
        recursive_choose_generator(all_cells_false, mines_to_set)
        return result

    def solve_cluster(self, remaining_mines):
        ''' Use CSP to find the solution to the CSP. Solution is the list of
        all possible mine/safe variations that fits all groups' condition.
        Solution is in the form of a list of Tru/False (Tru for mine,
        False for safe), with positions in the solution corresponding to cells
        in self.cells list. SOlution will be stored in self.solutions
        Will result in empty solution if the initial cluster is too big.
        '''
        # for clusters with 1 group - there is not enough data to solve them
        if len(self.groups) == 1:
            return

        # It gets too slow and inefficient when
        # - There are too many, but clusters but not enough groups
        #    (cells/clusters > 12)
        # - Long clusters (>40 cells)
        # - Groups with too many possible mine positions (> 1001)
        # Do not solve such clusters
        if len(self.cells_set) / len(self.groups) > 12 or \
           len(self.cells_set) > 50 or \
           max(math.comb(len(group.cells), group.mines)
                for group in self.groups) > 1001:
            return

        # We need to fix the order of cells, for that we populate self.cells
        self.cells = list(self.cells_set)

        # We also need a way to find a position of each cell by
        # the cell itself. So here's the addressing dict
        cells_positions = {cell: pos for pos, cell in enumerate(self.cells)}

        # The list to put all the solutions in.
        # Each "solution" is a list of [True, False, None, ...],
        # corresponding to cluster's ordered cells,
        # where True is a mine, False is safe and None is unknown
        # It starts with all None and will be updated for each group
        solutions = [[None for _ in range(len(self.cells))], ]

        for group in self.groups:

            # List of all possible ways mines can be placed in
            # group's cells: for example: [(False, True, False), ...]
            mine_positions = self.all_mines_positions(len(group.cells),
                                                      group.mines)
            # Now the same, but with cells as keys
            # For example: {cell1: False, cell2: True, cell3: False}
            mine_positions_dict = [dict(zip(group.cells, mine_position))
                                   for mine_position in mine_positions]

            # Failsafe: if there are more than 1_000_000 combinations
            # to go through - pull the plug
            if len(solutions) * len(mine_positions) > 1_000_000:
                return

            # This is where will will put solutions,
            # updated with current group's conditions
            updated_solutions = []

            # Go through each current solution and possible
            # mine distribution in a group
            for solution in solutions:
                for mine_position in mine_positions_dict:

                    updated_solution = solution.copy()

                    # Check if this permutation fits with this solution
                    for cell in group.cells:
                        # This is the position of this cell in the solution
                        position = cells_positions[cell]
                        # If solution has nothing about this cell,
                        # just update it with cell data
                        if updated_solution[position] is None:
                            updated_solution[position] = mine_position[cell]
                        # But if there is already mine or safe in the solution:
                        # it should be the same as in the permutation
                        # If it isn't: break to the next permutation
                        elif updated_solution[position] != mine_position[cell]:
                            break
                    # If we didn't break (solution and permutation fits),
                    # Add it to the next round
                    else:
                        updated_solutions.append(updated_solution)

            solutions = updated_solutions

        # Check if there are no more mines in solutions than remaining mines
        for solution in solutions:
            mine_count = sum(1 for mine in solution if mine)
            if mine_count <= remaining_mines:
                self.solutions.append(solution)

    def calculate_frequencies(self):
        ''' Once the solution is there, we can calculate frequencies:
        how often is a cell a mine in all solutions. Populates the
        self.frequencies with a dict {cell: frequency, ... },
        where frequency ranges from 0 (100% safe) to 1 (100% mine).
        Also, use weights  (self.solution_weights) - it shows in how many
        cases this solution is likely to appear.
        '''
        # Can't do anything if there are no solutions
        if not self.solutions:
            return

        for position, cell in enumerate(self.cells):
            count_mines = 0
            for solution_n, solution in enumerate(self.solutions):
                # Mine count takes into account the weight of teh solution
                # So if fact it is 1 * weight
                if solution[position]:
                    count_mines += self.solution_weights[solution_n]
            # Total in this case - not the number of solutions,
            # but weight of all solutions
            self.frequencies[cell] = count_mines / sum(self.solution_weights)

    def safe_cells(self):
        ''' Return list of guaranteed safe cells (0 in self.frequencies)
        '''
        safe = [cell for cell, freq in self.frequencies.items() if freq == 0]
        return safe

    def mine_cells(self):
        '''Return list of guaranteed mines (1 in self.frequencies)
        '''
        mines = [cell for cell, freq in self.frequencies.items() if freq == 1]
        return mines

    def calculate_hash(self):
        ''' Hash of a cluster. To check if we already dealt with this one
        '''
        # Prepare data for hashing: sort cells, add number of groups
        # (should be enough to identify a cluster)
        for_hash = sorted(list(self.cells_set)) + [len(self.groups)]
        # Make immutable
        for_hash = tuple(for_hash)
        # Return hash
        return hash(for_hash)

    def calculate_solution_weights(self, covered_cells, remaining_mines):
        ''' Calculate how probable each solution  is,
        if there are total_mines left in the field. That is,
        how many combinations of mines are possible with this solution.
        Populate self.solution_weights with the results
        '''
        self.solution_weights = []

        # For each solution we calculate how  many combination are possible
        # with the remaining mines on the remaining cells
        for solution in self.solutions:
            solution_mines = sum(1 for mine in solution if mine)
            solution_comb = math.comb(len(covered_cells) - len(solution),
                                      remaining_mines - solution_mines)
            self.solution_weights.append(solution_comb)

    def calculate_probable_mines(self):
        '''Based on solution and weights, calculate, how many mines we expect
        this cluster to have, on average.
        '''
        # Special case: cluster is made up of 1 group
        # Than the number of mines in this group  is the answer
        if len(self.groups) == 1:
            self.probable_mines = self.groups[0].mines
            return

        # Solved cluster
        solutions_mines = 0
        solutions_count = 0
        # Look at each solution
        for solution_n, solution in enumerate(self.solutions):
            # Number of mines would be equal to True's in solution
            # And weight shows how many times this solution can be repeated
            solutions_mines += sum(1 for position in solution if position) * \
                self.solution_weights[solution_n]
            solutions_count += self.solution_weights[solution_n]

        # If solution was not empty: return average sum of all mines
        # across all weighted solutions
        if solutions_count != 0:
            self.probable_mines = solutions_mines / solutions_count
            return

        # If cluster wasn't solved:
        cells_mine_probabilities = {}
        for group in self.groups:

            # Probability of mines for each cell in this group
            groups_probability = group.mines / len(group.cells)

            for cell in group.cells:
                # Put together a list of all probabilities
                if cell not in cells_mine_probabilities:
                    cells_mine_probabilities[cell] = []
                cells_mine_probabilities[cell].append(groups_probability)

        # Result is the sum of all averages
        total_mines = 0
        for cell, probabilities in cells_mine_probabilities.items():
            total_mines += sum(probabilities) / len(probabilities)
        self.probable_mines = total_mines

    def __str__(self):
        output = f"Cluster with {len(self.groups)} group(s) "
        output += f"and {len(self.cells_set)} cell(s): {self.cells_set}"
        return output


class MineProbability:
    '''Class to work with probability-based information about cells
    '''

    def __init__(self):
        # Probability of mines in this cell
        self.value = {}
        # Which method generated  this probability
        self.source = {}
        # Probability of getting a 0 in this cell
        self.opening_chance = {}

    def pick_lowest_probability(self):
        ''' Pick and return the cell(s) with the lowest mine probability,
        from the self.mine_probabilities.
        Also, return lowest probability predicted for this cell.
        '''
        lowest_probability = None
        highest_opening_chance = None
        least_likely_cells = []

        # Go through the probability dict
        for cell, probability in self.value.items():

            opening_chance = self.opening_chance[cell]

            # Re-start the list if:
            # - The list is empty
            # - We found lower mine probability
            # - Mine probability is the same, but opening probability is higher
            if lowest_probability is None or \
               probability < lowest_probability or \
               probability == lowest_probability and \
               opening_chance > highest_opening_chance:
                least_likely_cells = [cell, ]
                lowest_probability = probability
                highest_opening_chance = opening_chance

            # Or if it is the same probability - add to the list
            elif (probability == lowest_probability and
                  highest_opening_chance == opening_chance):
                least_likely_cells.append(cell)

        return least_likely_cells


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
        self.groups = AllMineGroups()

        # Placeholder for clusters (main element of CSP method)
        self.clusters = []

        # History of clusters we already processed, so we won't have to
        # solved them again (they can be quite time consuming)
        self.clusters_history = set()

        # Placeholder for mine probability data
        # {cell: probability_of_it_being_a_mine}
        self.probability = MineProbability()

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
                new_group = MineGroup(covered_neighbors, active_mines)
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
            new_cluster = CellCluster(self.groups.next_non_clustered_groups())

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

    def calculate_unaccounted(self):
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
        self.unaccounted_group = MineGroup(unaccounted_cells,
                                           unaccounted_mines)

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
                    new_group = MineGroup(group_b.cells - group_a.cells,
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
            if cluster.calculate_hash() in self.clusters_history:
                continue

            # Solve the cluster
            cluster.solve_cluster(self.remaining_mines)
            cluster.calculate_solution_weights(self.covered_cells,
                                               self.remaining_mines)
            cluster.calculate_frequencies()

            # Get safe cells and mines from cluster
            safe.extend(cluster.safe_cells())
            mines.extend(cluster.mine_cells())

            # Save cluster's hash in history for deduplication
            self.clusters_history.add(cluster.calculate_hash())

        return list(set(safe)), list(set(mines))

    def calculate_remaining_mines(self):
        ''' Based on mines we know about, calculate how many are still
        on the filed. Populate self.remaining_mines
        '''
        self.remaining_mines = self.total_mines
        for cell in self.helper.iterate_over_all_cells():
            if self.field[cell] == ms.CELL_MINE:
                self.remaining_mines -= 1

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
                self.probability.value[cell] = background_probability
                self.probability.source[cell] = "Background"

        def unaccounted_probabilities(self):
            ''' Background probability is all mines / all covered cells .
            It is quite crude and often inaccurate, but sometimes it is better
            to click deep into the unknown rather than try 50/50 guess.
            '''
            # It can be empty, and in that case we can't use it
            if self.unaccounted_group is None or \
               len(self.unaccounted_group.cells) == 0:
                return

            # Only use "perfect" coverage, meaning there should be no groups
            # that overlap with unaccounted group
            for group in self.groups:
                if self.unaccounted_group.cells.difference(group.cells):
                    return

            unaccounted_probability = \
                self.unaccounted_group.mines / \
                len(self.unaccounted_group.cells)
            for cell in self.unaccounted_group.cells:
                self.probability.value[cell] = unaccounted_probability
                self.probability.source[cell] = "Unaccounted"

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
                    if group_probability > self.probability.value[cell]:
                        self.probability.value[cell] = group_probability
                        self.probability.source[cell] = "Groups"

        def csp_probabilities(self):
            ''' Calculate mine possibilities based on CSP
            clusters
            '''
            for cluster in self.clusters:
                for cell, frequency in cluster.frequencies.items():
                    # Overwrite the probability result
                    self.probability.value[cell] = frequency
                    self.probability.source[cell] = "CSP"

        def cluster_leftovers_probabilities(self):
            ''' Estimate the mine numbers in clusters, and deduce the chance
            of mines outside of clusters. Basically, it is a more accurate
            version of background probabilities.
            '''
            for cluster in self.clusters:
                print(cluster)
                cluster.calculate_probable_mines()
                print(cluster.probable_mines)


        # Reset probabilities
        self.probability = MineProbability()
        # Background probability: all remaining mines on all covered cells
        background_probabilities(self)
        # Unaccounted (uncovered and not in groups) probability
        unaccounted_probabilities(self)
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
        for cell in self.probability.value:
            opening_chance = 1
            # Look at neighbors of each cell
            for neighbor in self.helper.cell_surroundings(cell):
                # If there are any mines around, there is no chance of opening
                if self.field[neighbor] == ms.CELL_MINE:
                    self.probability.opening_chance[cell] = 0
                    break
                # Otherwise each mine chance decrease opening chance
                # by (1 - mine chance) times
                if neighbor in self.probability.value:
                    opening_chance *= (1 - self.probability.value[neighbor])
            else:
                self.probability.opening_chance[cell] = opening_chance

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

    def solve(self, field):
        ''' Main solving function.
        Go through various solving methods and return safe and mines lists
        as long as any of the methods return results
        In:
        - the field (what has been uncovered so far).
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
        # Generate groups (main data for basic solving methods)
        self.generate_groups()
        # Number of remaining mines
        self.calculate_remaining_mines()
        # And a list of all covered cells
        self.generate_all_covered()
        # Unaccounted cells (covered - mines)
        self.calculate_unaccounted()

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

        # Calculate mine probability using various methods
        self.calculate_probabilities()
        # Calculate opening chances
        self.calculate_opening_chances()

        # Pick a cell that is least likely a mine
        lucky_cells = self.probability.pick_lowest_probability()

        if lucky_cells:
            lucky_cell = self.pick_a_random_cell(lucky_cells)
            self.last_move_info = ("Probability",
                                   self.probability.source[lucky_cell],
                                   self.probability.value[lucky_cell])
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

    game = ms.MinesweeperGame(settings, seed=0.660245378622389)
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
