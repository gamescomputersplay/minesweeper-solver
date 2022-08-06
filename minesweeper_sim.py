''' Class for simulating minesweeper games:
generte board, input a move, answer with resulting boards etc.
'''

import random

from dataclasses import dataclass
import numpy as np


@dataclass
class GameSettings:
    ''' Class to hold game settings:
    widyjs and height of teh board, number of mines
    '''
    width: int = 8
    height: int = 8
    mines: int = 10


# Presets for main game sizes
GAME_BEGINNER = GameSettings(8, 8, 10)
GAME_INTERMEDIATE = GameSettings(16, 16, 40)
GAME_EXPERT = GameSettings(30, 16, 99)
GAME_TEST = GameSettings(5, 4, 4)

# Cell types
CELL_MINE = -1
# Others are for the self.uncovered field
CELL_COVERED = -2  # Never clicked
CELL_FALSE_MINE = -3  # Marked mine, but actually isn't
CELL_EXPLODED_MINE = -4  # Explosion (clicked safe, but it was a mine)

# MinesweeperGame.status: returned by the do_move, tells the result of the move
STATUS_ALIVE = 0
STATUS_DEAD = 1
STATUS_WON = 2


class MinesweeperGame:
    ''' Class for a minesweeper game: game board, revealed game board,
    making a move etc
    '''

    def __init__(self, settings=GAME_BEGINNER, seed=None):
        ''' Initiate a new game: generate mines, calculate numbers
        '''

        # Unpack game parameters
        self.width = settings.width
        self.height = settings.height
        # Make sure there no more mines than cells
        self.mines = min(settings.mines, self.width * self.height)

        self.remaining_mines = self.mines

        # Use random seed, if passed
        if seed is not None:
            random.seed(seed)

        # Generate field: array of mines and numbers that player
        # cannot see yet
        self.field = self.generate_mines()
        # Populate it with numbers
        self.generate_numbers()

        # Initiate "uncovered": the part of the field
        # that player sees
        self.uncovered = np.full((self.width, self.height), CELL_COVERED)

        # Default status
        self.status = STATUS_ALIVE

    def iterate_over_all_cells(self):
        ''' Returns a list [(0,0), (0,1) .. (width-1, height-1)]
        kind of like "range" but for 2d array of cells.
        '''
        permutations = []
        # Iterate over self.height, so it would go line after line,
        # not column after column
        for j in range(self.height):
            for i in range(self.width):
                permutations.append((i, j))
        return permutations

    def valid_coords(self, cell):
        ''' Check if cell's coordinates are valid
        (do't go over field's size).
        Return Tru / False
        '''
        coord_x, coord_y = cell
        if 0 <= coord_x < self.width and 0 <= coord_y < self.height:
            return True
        return False

    def cell_surroundings(self, cell):
        ''' Returns a list of neighbours of a cell coord_x, coord_y,
        taking borders into account
        '''
        surroundings = []
        coord_x, coord_y = cell
        # Iterate over 3x3
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):

                # Not the cell itself
                if (i, j) != (0, 0):
                    # And is not outside the border
                    if self.valid_coords((coord_x + i, coord_y + j)):
                        surroundings.append((coord_x + i, coord_y + j))

        return surroundings

    def random_coords(self):
        ''' Generate a pair of random coordinates
        within field's size. Returns a tuple
        '''
        random_x = random.randint(0, self.width - 1)
        random_y = random.randint(0, self.height - 1)
        return (random_x, random_y)

    def generate_mines(self):
        '''Generate a game field (np array) with this size and mines
        according to the setting. Mines are marked as -1
        '''
        # Start with an empty NP array
        field = np.full((self.width, self.height), 0)

        # Set all mines
        for _ in range(self.mines):

            # Keep trying until a free cell is found
            while True:

                cell = self.random_coords()

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
        # Itrate over all cells
        for cell in self.iterate_over_all_cells():

            # We only need non-mines cells
            if self.field[cell] == CELL_MINE:
                continue

            # Iteration over neighbours of a mine
            # Add 1 for any mine neighbour
            for (neib_x, neib_y) in self.cell_surroundings(cell):
                if self.field[neib_x, neib_y] == CELL_MINE:
                    self.field[cell] += 1

    def has_covered(self):
        ''' The game still has covered cells.
        Used to detect stuck games
        '''
        for cell in self.iterate_over_all_cells():
            if self.uncovered[cell] == CELL_COVERED:
                return True
        return False

    def all_are_uncovered(self):
        ''' Are all cells covered?
        (to indicate the this is the very first move)
        '''
        for cell in self.iterate_over_all_cells():
            if self.uncovered[cell] != CELL_COVERED:
                return False
        return True

    def is_solved(self):
        ''' Check if the game is won.
        It is, if there are no covered cells, that cover non-mines.
        And whatever mines are marked are actually mines
        (this should work for both normal and non-flag play styles)
        '''
        for cell in self.iterate_over_all_cells():
            if (self.uncovered[cell] == CELL_COVERED or
                    self.uncovered[cell] == CELL_MINE) and \
                    self.field[cell] != CELL_MINE:
                return False

        return True

    def reveal_uncovered(self):
        ''' When the game is lost, we show all mines
        and wrongly marked mines on self.uncovered
        '''
        for cell in self.iterate_over_all_cells():
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
            for neighbour in self.cell_surroundings(current_zero):

                # Uncover any covered cells
                if self.uncovered[neighbour] == CELL_COVERED:
                    self.uncovered[neighbour] = self.field[neighbour]

                    # If it happen to be a 0: add it to the zero list
                    if self.field[neighbour] == 0:
                        zeros.append(neighbour)

    def safe_first_click(self, cell):
        ''' Check conditions for teh first click.
        If it is a mine, move it. Recalculate field.
        '''
        if self.valid_coords(cell) and self.all_are_uncovered() and \
           self.field[cell] == CELL_MINE:

            while True:

                move_to_cell = self.random_coords()

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
        if not self.valid_coords(cell):
            return

        if self.uncovered[cell] == CELL_COVERED:
            self.uncovered[cell] = CELL_MINE
            self.remaining_mines -= 1

    def handle_safe_click(self, cell):
        ''' Safe (left) click. Explode if a  mine,
        uncover if not. (includes flood fill for 0)
        '''
        # Ignore invalid coordinates
        if not self.valid_coords(cell):
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

    def field2str(self, field_to_show, show_ruler=True):
        ''' Visual representation of the board
        (Can be used for both secret and uncovered boards)
        '''
        # Characters to show for different statuses
        characters_to_display = {**{
            CELL_MINE: "*",
            CELL_COVERED: " ",
            CELL_FALSE_MINE: "X",
            CELL_EXPLODED_MINE: "!",
            0: "."
        }, **{i: str(i) for i in range(1, 9)}}

        # Add all data to this string
        output = ""

        # Top ruler + shift fot the top border
        if show_ruler:
            output += " " * 5
            # Top ruler shows only second digit for numbers ending 1..9
            # but both digits for those ending in 0 etc
            output += "".join([f"{i}" if i % 10 == 0 and i != 0
                               else f"{i % 10} "
                               for i in range(self.width)])
            output += "\n"
            output += " " * 3

        # Top border
        output += "-" * (self.width * 2 + 3) + "\n"

        # Iterate over all cells
        for row in range(self.height):

            # Left border
            if show_ruler:
                output += f"{row:2} "
            output += "! "

            for col in range(self.width):

                cell_value = field_to_show[col, row]

                # Display the character according to characters_to_display
                output += characters_to_display[cell_value] + " "

            # Right border
            output += "!\n"

        # Bottom border
        if show_ruler:
            output += " " * 3
        output += "-" * (self.width * 2 + 3) + "\n"

        return output

    def __str__(self):
        ''' Display uncovered part of the board
        '''
        return self.field2str(self.uncovered)

    def parse_input(self, string):
        ''' For Command line play: parce string as
        [x y] | [M x y] | [A x y] ...
        - where x and y are 0-based coordinates
        - "M" to mark the mine
        - "A" is open all around (like 2-button click)
        returns two lists that can be fed right into "make_a_move"
        '''
        safe, mines = [], []
        mode, coord_x, coord_y = None, None, None

        # Break th estring by spaces
        for chunk in string.upper().split(" "):

            # We have a coordinate
            if chunk.isnumeric():
                # First one must be an X, second an Y
                if coord_x is None:
                    coord_x = int(chunk)
                else:
                    coord_y = int(chunk)
                    cell = (coord_x, coord_y)

                    # Append the lists accordingly
                    # Single mine
                    if mode == "M":
                        mines.append(cell)
                    # Open around, add all surroudings
                    elif mode == "A":
                        safe.extend(self.cell_surroundings(cell))
                    # Single safe
                    else:
                        safe.append(cell)
                    mode, coord_x, coord_y = False, None, None

            # M and A modify the behaviour
            elif chunk in ("M", "A"):
                mode = chunk

        return safe, mines


def main():
    ''' Some tests for minesweeper sim
    '''

    # Random seed to generatethe game
    seed = None

    status_message = {STATUS_ALIVE: "Still alive",
                      STATUS_DEAD: "You died",
                      STATUS_WON: "You won"}

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
        print(f"Status: {status_message[game.status]}")
        print(f"Remaining mines: {game.remaining_mines}")

    print("Thanks you for playing!")

if __name__ == "__main__":
    main()
