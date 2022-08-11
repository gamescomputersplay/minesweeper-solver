''' Class for the solver of minesweeper game.
Game mechanics and some helper functions are taken from
minesweeper-game.py
'''

import random
import math
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

        # Group type ("exactly", "no more than", "at least")
        self.group_type = group_type

        # Calculate hash (for dedupliaction)
        self.hash = self.calculate_hash()

        # Placeholder for cluster information
        # (each group can belong to only one cluster)
        self.belongs_to_cluster = None

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
    ''' CellCluster is a group of cells connected together by an overlaping
    list of groups. In other words mine/safe in any of the cell, can
    potentially trigger safe/mine in any other cell of the cluster.
    Is a basic class for CSP (constrait satisfaction problem) method
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
        ''' Generate all permuations for "Choose k from n",
        which is equivalent to all possible ways mines are
        located in the cells.
        Result is a list of tuples like (False, False, True, False),
        indicating if the item was chosen (if there is a mine in the cell).
        For example, for generate_mines_permutations(2, 1) the output is:
        [(True, False), (Fasle, True)]
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

    def solve_csp(self):
        ''' Use CSP to find safe cells and mines in the cluster
        '''
        # It gets too slow and inefficient when
        # - There are too many, but clusters but not enough groups
        #    (cells/clusters > 12)
        # - Long clusters (>40 cells)
        # - Groups with too many possible mine positions (> 1001)
        # Do not solve such clusters
        if len(self.cells_set) / len(self.groups) > 12 or \
           len(self.cells_set) > 40 or \
           max([math.comb(len(group.cells), group.mines)
                for group in self.groups]) > 1001:
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

        self.solutions = solutions

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

    def __str__(self):
        output = f"Cluster with {len(self.groups)} group(s) "
        output += f"and {len(self.cells_set)} cell(s): {self.cells_set}"
        return output


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

        # Placeholder for clusters (main element of CSP method)
        self.clusters = []

        # History of clusters we already processed, so we won't have to
        # solved them again (they can be quite time consuming)
        self.clusters_history = set()

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

            # Groups are only for numbered cells
            if self.field[cell] <= 0:
                continue

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

    def generate_clusters(self):
        '''Populate self.clsters with cell clusters
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
                    # If it overlanps with this group and not part of any
                    # other cluster - add this group
                    if group.group_type == "exactly" and \
                       group.belong_to_cluster is None and \
                       new_cluster.overlap(group):
                        new_cluster.add(group)
                        break

                # We went through the groups without adding any:
                # new_cluster is done
                else:
                    # We don't want clusters made of 1 group
                    if len(new_cluster.groups) > 1:
                        self.clusters.append(new_cluster)
                    # But exit the while anyway
                    break

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
        # Generate clusters
        self.generate_clusters()
        for cluster in self.clusters:

            # Cluster deduplication. If we solved this cluster already, ignore
            if cluster.calculate_hash() in self.clusters_history:
                continue

            # Solve the cluster
            # print(f"CSP! Cluster: {cluster}")
            cluster.solve_csp()

            # Save cluster's hash in history for deduplication
            self.clusters_history.add(cluster.calculate_hash())

        safe, mines = [], []
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
        safe, mines = self.method_subgroups()
        if safe or mines:
            return safe, mines

        # 4. CSP method
        # (stands for Constraint Satisfaction Problem)
        ######################
        safe, mines = self.method_csp()
        if safe or mines:
            return safe, mines

        # If there is nothing we can do: return a random cell
        return [self.pick_a_random_cell(covered_cells), ], None


def main():
    ''' Test the solver on a simple game
    '''

    settings = ms.GAME_TEST
    settings = ms.GAME_BEGINNER

    game = ms.MinesweeperGame(settings, seed=0)
    solver = MinesweeperSolver(settings)

    while game.status == ms.STATUS_ALIVE:

        safe, mines = solver.solve(game.uncovered)
        print(f"Move: Safe {safe}, Mines {mines}")
        game.make_a_move(safe, mines)
        print(game)

    print(ms.STATUS_MESSAGES[game.status])


if __name__ == "__main__":
    main()
