''' Some classes for Minesweeper  solver (minesweeper_solver.py)
'''

import math
from dataclasses import dataclass

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


class AllGroups:
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


class GroupCluster:
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

        # Dict of possible mine counts {mines: mine_count, ...}
        self.probable_mines = {}

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
            if sum(self.solution_weights) > 0:
                self.frequencies[cell] = count_mines / \
                    sum(self.solution_weights)
            # This shouldn't normally happen, but it may rarely happen during
            # "next move" method, when "Uncovered but unknown" cell is added
            else:
                self.frequencies[cell] = 0

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

    def possible_mine_counts(self):
        ''' Based on solution and weights, calculate a dict with possible
        mine counts. For example, {3: 4, 2: 1},  4 solutions with 3 mines,
        1 solution with 2. Put it in self.probable_mines
        Will be used for CSP Leftovers probability calculation.
        '''

        # Cluster was solved
        if self.solutions:
            # Look at each solution
            for solution in self.solutions:
                mines_count = sum(1 for position in solution if position)
                if mines_count not in self.probable_mines:
                    self.probable_mines[mines_count] = 0
                self.probable_mines[mines_count] += 1
            return

        # If cluster wasn't solved (which is basically never happens
        # on 2D field, but can happen at higher dimensions):
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
        self.probable_mines[int(total_mines)] = 1

    def __str__(self):
        output = f"Cluster with {len(self.groups)} group(s) "
        output += f"and {len(self.cells_set)} cell(s): {self.cells_set}"
        return output


class AllClusters:
    ''' Class that holds all clusters and leftovers data
    '''

    def __init__(self):
        # List of all clusters
        self.clusters = []

        # History of clusters we already processed, so we won't have to
        # solved them again (they can be quite time consuming)
        # Also used to cache mine counts, {hash_value: mine_count}
        self.clusters_history = {}

        # Cells that are not in any cluster
        self.leftover_cells = set()

        # Count and weights of mines in leftover cells. For example:
        # {3:20, 4:50} - there are 20 ways it can be 3 mines and 50 ways
        # it can be 4 mines
        self.leftover_weights = {}

        # Average chance of a mine in leftover cells (None if NA)
        self.leftover_mine_chance = None

    def reset(self):
        ''' CLear the list of clusters, but not the cluster history
        '''
        self.clusters = []
        self.leftover_cells = set()
        self.leftover_weights = {}
        self.leftover_mine_chance = None

    def calculate_all(self, covered_cells, remaining_mines):
        '''Perform all cluster-related calculations: solve clusters,
        calculate weights and mine frequencies etc. Check the history
        in case this cluster was already solved
        '''
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
            cluster.solve_cluster(remaining_mines)
            cluster.calculate_solution_weights(covered_cells,
                                               remaining_mines)
            cluster.calculate_frequencies()
            cluster.possible_mine_counts()

            # Save cluster's hash in history for deduplication
            self.clusters_history[cluster.calculate_hash()] = \
                cluster.probable_mines

    def calculate_leftovers(self, covered_cells, remaining_mines):
        '''Based on count and probabilities of mines in cluster, calculate
        mines and probabilities in cells that don't belong to any clusters.
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
        self.leftover_cells = set(covered_cells).\
            difference(cells_in_clusters)

        # If no unaccounted cells - can't do anything
        if not self.leftover_cells:
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
        self.leftover_weights = {mines: math.comb(len(self.leftover_cells),
                                remaining_mines - mines) * solutions
                                for mines, solutions in cross_mines.items()
                                if remaining_mines - mines >= 0}

        # Total weights for all mine counts
        total_weights = sum(self.leftover_weights.values())

        # If the cluster was not solved total weight would be zero.
        if total_weights == 0:
            return

        # Total number of mines in all clusters
        # (sum of count * probability)
        mines_in_clusters = sum(mines * weight / total_weights
                                for mines, weight
                                in self.leftover_weights.items())

        # And this is the probability of a mine in those cells
        self.leftover_mine_chance = (remaining_mines - mines_in_clusters) \
            / len(self.leftover_cells)


@dataclass
class CellProbabilityInfo:
    '''Data about mine probability for one cell
    '''
    # Chance this cell is a mine (0 to 1)
    mine_chance: float
    # Which method was used to generate mine chance (for statistics)
    source: str
    # Chance it would be an opening (no mines in surrounding cells)
    opening_chance: float = 0
    # Number of solved cells in the next round, if this cell is uncovered
    next_sure_clicks: int = 0


class AllProbabilities(dict):
    '''Class to work with probability-based information about cells
    '''

    def pick_lowest_probability(self):
        ''' Pick and return the cell(s) with the lowest mine probability,
        and, if several, with highest opening probability.
        '''
        # Copy the info into a list, so we can just sort it
        cells = [(cell, cell_info.mine_chance, cell_info.opening_chance)
                 for cell, cell_info in self.items()]

        # Sort by 1. mine chance 2. opening chance. Put the best at the end
        cells.sort(key=lambda x: (x[1], -x[2]))

        # This is the best chances
        _, best_mine_chance, best_opening_chance = cells[0]
        cells_best_chances = []

        # Pick cells with the best probability and opening chances
        for cell, mine_chance, opening_chance in cells:
            if mine_chance == best_mine_chance and \
               opening_chance == best_opening_chance:
                cells_best_chances.append(cell)

        # Return cells with lowest mine chance and highest open chance.
        # Later we'll add a logic to look playing "Future" boards to determine
        # survivability over 2 and more moves. Not right now though.
        return cells_best_chances


def main():
    ''' Display message that this is not a standalone file
    '''
    print("This file contains some classes for minesweeper.py")
    print("There is no code to run in __main__ mode")


if __name__ == "__main__":
    main()