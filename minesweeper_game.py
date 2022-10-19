''' Class for simulating n-dimensional minesweeper games:
generate board, input a move, answer with resulting boards etc.
(Visualization only works for n in (2, 3, 4))
'''

import random

from dataclasses import dataclass
import numpy as np


@dataclass
class GameSettings:
    ''' Class to hold game settings:
    dimensions of the board, number of mines, use wrap around
    '''
    shape: tuple = (8, 8)
    mines: int = 10
    density: float = 0
    wrap_around: bool = False

    def __post_init__(self):
        # Calculate mine density (used only for information)
        self.density = self.mines / np.prod(self.shape)

    def __str__(self):
        output = ""
        output += f"Dims:{len(self.shape)}, "
        output += f"Shape:{self.shape}, "
        output += f"Volume:{np.prod(self.shape)}, "
        output += f"Mines:{self.mines}, "
        output += f"Density:{self.density:.1%}, "
        output += f"Wrap:{'yes' if self.wrap_around else 'no'}, "
        return output


# Presets for main game sizes
# Classical minesweeper difficulties
GAME_BEGINNER = GameSettings((8, 8), 10)
GAME_INTERMEDIATE = GameSettings((16, 16), 40)
GAME_EXPERT = GameSettings((30, 16), 99)

# 3D examples
GAME_3D_EASY = GameSettings((5, 5, 5), 10)
GAME_3D_MEDIUM = GameSettings((7, 7, 7), 33)
GAME_3D_HARD = GameSettings((10, 10, 10), 99)

# 4D examples
GAME_4D_EASY = GameSettings((4, 4, 4, 4), 10)
GAME_4D_MEDIUM = GameSettings((4, 4, 4, 4), 20)
GAME_4D_HARD = GameSettings((4, 4, 4, 4), 30)
GAME_4D_HARDER = GameSettings((4, 4, 4, 4), 40)

# Classic games, but with wrap around
GAME_BEGINNER_WRAP = GameSettings((8, 8), 10, wrap_around=True)
GAME_INTERMEDIATE_WRAP = GameSettings((16, 16), 40, wrap_around=True)
GAME_EXPERT_WRAP = GameSettings((30, 16), 99, wrap_around=True)

# Exotic settings
GAME_1D = GameSettings((60,), 10)
GAME_6D = GameSettings((4, 4, 4, 4, 4, 4), 50)

# Small board for testing
GAME_TEST = GameSettings((5, 4), 4)

# Cell types
CELL_MINE = -1
# Others are for the self.uncovered field
# Never clicked
CELL_COVERED = -2
# Marked mine, but actually isn't
CELL_FALSE_MINE = -3
# Explosion (clicked safe, but it was a mine)
CELL_EXPLODED_MINE = -4

# Characters to show for different statuses
# 80 is the highest possible number of neighbors (in a 4d game)
LEGEND = {**{
    CELL_MINE: "*",
    CELL_COVERED: " ",
    CELL_FALSE_MINE: "X",
    CELL_EXPLODED_MINE: "!",
    0: "."
}, **{i: str(i) for i in range(1, 81)}}

# MinesweeperGame.status: returned by the do_move, tells the result of the move
STATUS_ALIVE = 0
STATUS_DEAD = 1
STATUS_WON = 2
STATUS_MESSAGES = {STATUS_ALIVE: "Still alive",
                   STATUS_DEAD: "You died",
                   STATUS_WON: "You won"}


class MinesweeperHelper:
    ''' Class wth a few helper method to handle n-dimensional
    minesweeper boards: list of neighbors for each cell,
    iteration over all cells etc
    '''

    def __init__(self, shape, wrap_around=False):
        ''' Shape: list of sizes for dimensions (length of the list
        will define how many dimensions the game has)
        Wrap around: if opposite ends of the field are wrapped around
        (Surface of a torus for 2D game)
        '''
        self.shape = shape

        self.wrap_around = wrap_around

        # This is just a dict to store all the neighboring coordinates
        # for all cells, so we won't have to recalculate them every move
        self.neighbors_cache = {}

        # Cache for the list of all iterations
        self.all_iterations_cache = None

    def iterate_over_all_cells(self):
        ''' Returns a list
        [(0, .., 0), (0, .., 1) .. (d1-1, .., dn-1)]
        of all possible coordinates.
        Kind of like "range" but for D-dimensional array of cells.
        '''
        # Serve from cache, if available
        if self.all_iterations_cache is not None:
            return self.all_iterations_cache

        permutations = []

        # We'll have as many permutations as product of all dimensions
        for permutation_number in range(np.prod(self.shape)):

            # List to construct a permutation in
            this_permutation = []
            # And a copy of permutation's cardinal number
            # (need a copy, as we will mutate  it)
            remaining_number = permutation_number

            # Go from back to front through the dimension
            for pos in reversed(self.shape):

                # It is somewhat similar to base conversion,
                # except base changes for each place.

                # This permutation is just a remainder of division
                this_permutation.append(remaining_number % pos)
                # But you need to make sure you subtract by
                # the number in the latest position
                remaining_number -= this_permutation[-1]
                # and divide by the latest base
                remaining_number //= pos

            # Reverse the resulting list (as we started from the right side,
            # the smallest digits), and store it in the final list
            this_permutation.reverse()
            permutations.append(tuple(this_permutation))

        self.all_iterations_cache = permutations
        return permutations

    def valid_coords(self, cell):
        ''' Check if cell's coordinates are valid
        (do't go over field's size).
        Return Tru / False
        '''
        for i, dimension_size in enumerate(self.shape):
            if cell[i] < 0 or cell[i] >= dimension_size:
                return False
        return True

    def cell_surroundings(self, cell):
        ''' Returns a list of coordinates of neighbors of a cell
        taking borders into account
        '''
        # Dynamic programming: use buffer if the result is there
        if cell in self.neighbors_cache:
            return self.neighbors_cache[cell]

        surroundings = []

        # This is to calculate offset. But done outside of for
        # as this is the same for all iterations
        powers = {j: 3**j for j in range(len(self.shape))}

        # Iterate over 3 ** 'N of dimensions' of potential neighbors
        for i in range(3 ** len(self.shape)):

            # Way to calculate all (1, -1, 0) for this permutation
            offset = tuple((i // powers[j]) % 3 - 1
                           for j in range(len(self.shape)))

            # If it is the cell itself - skip
            if offset.count(1) == 0 and offset.count(-1) == 0:
                continue

            # Different ways of calculating neighbors' coordinates
            # with and without wrapping around
            if self.wrap_around:
                cell_with_offset = tuple((cell[i] + offset[i] +
                                          self.shape[i]) % self.shape[i]
                                          for i in range(len(self.shape)))
            else:
                cell_with_offset = tuple(cell[i] + offset[i]
                                         for i in range(len(self.shape)))

            # If resulting coords are valid: add them to the list of neighbors
            if self.valid_coords(cell_with_offset):
                surroundings.append(cell_with_offset)

        # Store in buffer, for future reuse
        self.neighbors_cache[cell] = surroundings

        return surroundings

    def random_coords(self):
        ''' Generates a tuple of self.size random coordinates
        each within the appropriate dimension's size. Returns a tuple
        '''
        coordinates = []
        for dimension_size in self.shape:
            coordinates.append(random.randint(0, dimension_size - 1))
        return tuple(coordinates)

    def are_all_covered(self, field):
        ''' Are all cells covered?
        (to indicate the this is the very first move)
        '''
        for cell in self.iterate_over_all_cells():
            if field[cell] != CELL_COVERED:
                return False
        return True


class MinesweeperGame:
    ''' Class for a minesweeper game: generate game board,
    accept a move, revealed the result, etc
    '''

    def __init__(self, settings=GAME_BEGINNER, seed=None, field_str=None):
        ''' Initiate a new game: generate mines, calculate numbers.
        Inputs:
        - settings: GameSettings objects with dimensions of the game
                    an the number of mines
        - seed: Seed to use for generating board. None for random.
        - field_str: pre-generated board to use. String with "*" for mines
        '''

        # Shape, a tuple of dimension sizes
        self.shape = settings.shape

        # Make sure there no more mines than cells minus one
        self.mines = min(settings.mines, np.prod(self.shape) - 1)

        # Wrap around option
        self.wrap_around = settings.wrap_around

        # Counter of remaining mines (according to what plays
        # marked as mines, can be inaccurate).
        self.remaining_mines = self.mines

        # Use seed, if seed passed
        if seed is not None:
            random.seed(seed)

        self.helper = MinesweeperHelper(self.shape, wrap_around=self.wrap_around)

        # Now we initiate "field": array of mines and numbers that player
        # cannot see yet

        # If field_str passed in - use it
        if field_str is not None:
            self.field = self.import_field(field_str)
        # Otherwise, generate the field
        else:
            self.field = self.generate_mines()

        # Populate it with numbers
        self.generate_numbers()

        # Initiate "uncovered": the part of the field
        # that player sees
        self.uncovered = np.full(self.shape, CELL_COVERED)

        # Default status
        self.status = STATUS_ALIVE

    def generate_mines(self):
        '''Generate a game field (np array) with this size and mines
        according to the setting. Mines are marked as -1
        '''
        # Start with an 0-filled NP array
        field = np.full(self.shape, 0)

        # Set all mines
        for _ in range(self.mines):

            # Keep trying until a free cell is found
            while True:

                cell = self.helper.random_coords()

                # If the spot is empty: set the mine there
                if field[cell] == 0:
                    field[cell] = CELL_MINE
                    # Move to the next mine
                    break

        return field

    def generate_numbers(self):
        ''' Calculate numbers for the self.field
        Will be used when field is generated or if a mine was moved
        (because of a non-mine first move)
        '''
        # Iterate over all cells
        for cell in self.helper.iterate_over_all_cells():

            # We only need non-mines cells
            if self.field[cell] == CELL_MINE:
                continue

            # Resent the number (in case it is a recount)
            self.field[cell] = 0

            # Iteration over neighbors of a mine
            # Add 1 for any mine neighbor
            for neighbor in self.helper.cell_surroundings(cell):
                if self.field[neighbor] == CELL_MINE:
                    self.field[cell] += 1

    def has_covered(self):
        ''' The game still has covered cells.
        Used to detect stuck games
        '''
        for cell in self.helper.iterate_over_all_cells():
            if self.uncovered[cell] == CELL_COVERED:
                return True
        return False

    def is_solved(self):
        ''' Check if the game is won.
        It is, if there are no covered cells, that cover non-mines.
        And whatever mines are marked are actually mines
        (this should work for both normal and non-flag play styles)
        '''
        for cell in self.helper.iterate_over_all_cells():
            if (self.uncovered[cell] == CELL_COVERED or
                    self.uncovered[cell] == CELL_MINE) and \
                    self.field[cell] != CELL_MINE:
                return False

        return True

    def reveal_uncovered(self):
        ''' When the game is lost, we show all mines
        and wrongly marked mines on self.uncovered
        '''
        for cell in self.helper.iterate_over_all_cells():
            # There is a mine that hasn't been revealed yet
            if self.field[cell] == CELL_MINE and \
               self.uncovered[cell] == CELL_COVERED:
                self.uncovered[cell] = CELL_MINE
            # There is a marked mine, that is not a mine
            if self.field[cell] != CELL_MINE and \
               self.uncovered[cell] == CELL_MINE:
                self.uncovered[cell] = CELL_FALSE_MINE

    def expand_zeros(self, cell):
        ''' User clicked on a cell "cell" that contained a zero
        Flood-fill open the cells around that zero
        '''
        zeros = [cell]

        while zeros:

            current_zero = zeros.pop()
            for neighbor in self.helper.cell_surroundings(current_zero):

                # Uncover any covered cells
                if self.uncovered[neighbor] == CELL_COVERED:
                    self.uncovered[neighbor] = self.field[neighbor]

                    # If it happen to be a 0: add it to the zero list
                    if self.field[neighbor] == 0:
                        zeros.append(neighbor)

    def safe_first_click(self, cell):
        ''' Check conditions for teh first click.
        If it is a mine, move it. Recalculate field.
        '''
        if self.helper.valid_coords(cell) and \
           self.helper.are_all_covered(self.uncovered) and \
           self.field[cell] == CELL_MINE:

            while True:

                move_to_cell = self.helper.random_coords()

                # If the spot is empty
                if self.field[move_to_cell] != CELL_MINE:
                    # move the mine there
                    self.field[move_to_cell] = CELL_MINE
                    # Clear the mine from wherever it was
                    self.field[cell] = 0
                    # Recalculate the numbers
                    self.generate_numbers()
                    # Break out of while
                    break

    def handle_mine_click(self, cell):
        ''' Mine (right) click. Simply mark cell a mine
        '''
        # Ignore invalid coordinates
        if not self.helper.valid_coords(cell):
            return

        if self.uncovered[cell] == CELL_COVERED:
            self.uncovered[cell] = CELL_MINE
            self.remaining_mines -= 1

    def handle_safe_click(self, cell):
        ''' Safe (left) click. Explode if a  mine,
        uncover if not. (includes flood fill for 0)
        '''
        # Ignore invalid coordinates
        if not self.helper.valid_coords(cell):
            return

        # This cell has been clicked on before: ignore it
        if self.uncovered[cell] != CELL_COVERED:
            return

        # There is a mine: it explodes, player dies
        if self.field[cell] == CELL_MINE:
            self.status = STATUS_DEAD
            # Mark the mine that killed player
            self.uncovered[cell] = CELL_EXPLODED_MINE

        # There is a number: reveal it
        else:
            self.uncovered[cell] = self.field[cell]
            # Do a flood-fill expansion, if a zero is opened
            if self.uncovered[cell] == 0:
                self.expand_zeros(cell)

    def make_a_move(self, safe=None, mines=None):
        ''' Do one minesweeper iteration.
        Accepts list of safe clicks and mine clicks.
        Returns self.status: 0 (not dead), 1 (won), 2 (dead)
        '''

        if safe:
            # Safe first click
            cell = safe[0]
            self.safe_first_click(cell)

            # Open all the safe cells
            # handle_safe_click Will set self.status = STATUS_DEAD
            # if clicked on a bomb
            for cell in safe:
                self.handle_safe_click(cell)

        if mines:
            # Mark all the mines
            for cell in mines:
                self.handle_mine_click(cell)

        # Stuck (marked more mines than there actually are)
        # Treat it as DEAD
        if not self.has_covered() and not self.is_solved():
            self.status = STATUS_DEAD

        # Won
        if self.is_solved():
            self.status = STATUS_WON

        # If DEAD or WON - reveal the board
        if self.status in (STATUS_DEAD, STATUS_WON):
            self.reveal_uncovered()
            self.remaining_mines = 0

        return self.status

    def field2str_1d(self, field_to_show, show_ruler=True):
        ''' Visual representation of the 1D field
        Done by converting it to a 2D field with height 1
        '''
        width = field_to_show.shape[0]
        field_to_show = np.reshape(field_to_show, (width, 1))

        return self.field2str_2d(field_to_show, show_ruler=True)

    @staticmethod
    def field2str_2d(field_to_show, show_ruler=True):
        ''' Visual representation of the 2D field
        '''
        height = field_to_show.shape[1]
        width = field_to_show.shape[0]

        # Add all data to this string
        output = ""

        # Top ruler + shift fot the top border
        if show_ruler:
            output += " " * 5
            # Top ruler shows only second digit for numbers ending 1..9
            # but both digits for those ending in 0 etc
            output += "".join([f"{i}" if i % 10 == 0 and i != 0
                               else f"{i % 10} "
                               for i in range(width)])
            output += " \n"
            output += " " * 3

        # Top border
        output += "-" * (width * 2 + 3) + "\n"

        # Iterate over all cells, row by row
        for row in range(height):

            # Left border
            if show_ruler:
                output += f"{row:2} "
            output += "! "

            # Iterate over each cell in a row
            for col in range(width):

                cell_value = field_to_show[col, row]

                # Display the character according to characters_to_display
                output += LEGEND[cell_value] + " "

            # Right border
            output += "!\n"

        # Bottom border
        if show_ruler:
            output += " " * 3
        output += "-" * (width * 2 + 3) + "\n"

        return output

    def field2str_3d(self, field_to_show, show_ruler=True):
        ''' Visual representation of the 3D field
        '''
        depth = field_to_show.shape[0]
        buffer = []
        # Slice 3D array into 2D arrays
        # And get strings for each one into a buffer
        for i in range(depth):
            buffer.append(self.field2str_2d(field_to_show[i])
                          .split("\n"))

        output = ""

        # Horizontal rule numbering each 2D field
        if show_ruler:

            # Pad it, so the number is in the middle of a field
            field_with = len(buffer[0][0])
            padding_left = " " * ((field_with - 2) // 2 + 2)
            padding_right = " " * (field_with - len(padding_left) - 1)

            for field_n, _ in enumerate(buffer):
                output += padding_left + str(field_n) + padding_right
            output += "\n"

        # Iterate over each line in the buffer.
        # Sort of "transpose" it
        for line_n in range(len(buffer[0])):
            for field in buffer:
                output += field[line_n]
            # We don't need the very last new line
            if line_n != len(buffer[0]) - 1:
                output += "\n"

        return output

    def field2str_4d(self, field_to_show, show_ruler=True):
        ''' Visual representation of the 4D field
        '''
        fourth = field_to_show.shape[0]
        output = ""
        for i in range(fourth):

            # Show ruler only for the first row
            show_ruler = i == 0

            # Get the 3D representation for the current row
            fields = self.field2str_3d(field_to_show[i],
                                       show_ruler=show_ruler)

            # Go through the 3D line by line
            cur_line_of_fields = fields.split("\n")[:-1]
            for line_n, line in enumerate(cur_line_of_fields):

                # Inserting either a row number or space
                if line_n == len(cur_line_of_fields) // 2 + \
                   (1 if i == 0 else 0):
                    output += str(i)
                else:
                    output += " "
                output += line + "\n"

        return output

    def field2str(self, field_to_show, show_ruler=True):
        ''' Pick the right string representation function:
        works for 2 to 4 D minesweeper fields
        '''
        if len(field_to_show.shape) == 1:
            return self.field2str_1d(field_to_show, show_ruler)
        if len(field_to_show.shape) == 2:
            return self.field2str_2d(field_to_show, show_ruler)
        if len(field_to_show.shape) == 3:
            return self.field2str_3d(field_to_show, show_ruler)
        if len(field_to_show.shape) == 4:
            return self.field2str_4d(field_to_show, show_ruler)
        return "Can't display the field, sorry"

    def __str__(self):
        ''' Display uncovered part of the board
        '''
        return self.field2str(self.uncovered)

    def parse_input(self, string):
        ''' For Command line play: parse string as
        [x y] | [M x y] | [A x y] ...
        - where x and y are 0-based coordinates
        - "M" to mark the mine
        - "A" is open all around (like 2-button click)
        returns two lists that can be fed right into "make_a_move"
        '''
        safe, mines = [], []
        mode, cell = None, []

        # Break the string by spaces
        for chunk in string.upper().split(" "):

            # We have a coordinate
            if chunk.isnumeric():
                # Add it to the coordinates list
                cell.append(int(chunk))

                # if coords is long enough
                if len(cell) == len(self.shape):

                    # It supposed to be a tuple
                    cell = tuple(cell)

                    # Append the lists accordingly
                    # Single mine
                    if mode == "M":
                        mines.append(cell)
                    # Open around, add all surroundings
                    elif mode == "A":
                        safe.extend(self.helper.cell_surroundings(cell))
                    # Single safe
                    else:
                        safe.append(cell)
                    mode, cell = False, []

            # M and A modify the behavior
            elif chunk in ("M", "A"):
                mode = chunk

        return safe, mines

    def export_field(self, field=None):
        '''Save the field into a log file, so later we will be able to import it.
        For debugging purposes
        '''
        # If no field passed in - use current game
        if field is None:
            field = self.field
        field_str = ""
        for cell in self.helper.iterate_over_all_cells():
            field_str += LEGEND[self.field[cell]]
        return field_str

    def import_field(self, field_str):
        '''Generate field from the string (that was saved by export)
        '''
        field = np.full(self.shape, 0)
        string_pointer = 0
        # Go through all cells and all characters in the field_str
        for cell in self.helper.iterate_over_all_cells():
            # Whenever field_str has "*" - it's a mine
            if field_str[string_pointer] == "*":
                field[cell] = CELL_MINE
            string_pointer += 1
        return field


def main():
    ''' Some tests for minesweeper sim
    '''

    # Seed to generate the game (None for random)
    seed = None

    game = MinesweeperGame(settings=GAME_TEST, seed=seed)

    # For debugging: check out the field
    # print(game.field2str(game.field))

    # Keep making moves, while alive
    while game.status == STATUS_ALIVE:

        # Input the move data
        string = input("Move: ")
        safe, mines = game.parse_input(string)
        game.make_a_move(safe, mines)

        # Display the results
        print(game)
        print(f"Status: {STATUS_MESSAGES[game.status]}")
        print(f"Remaining mines: {game.remaining_mines}")

    print("Thanks you for playing!")


if __name__ == "__main__":
    main()
