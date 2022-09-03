''' Bot that plays minesweeper "from pixels", including finding the game
field on the screen, recognizing the numbers, clicking cells etc.
Uses minesweeper solver in minesweeper_solver.py
'''

import time

# import pyautogui
from PIL import Image, ImageDraw
import numpy as np

import minesweeper_game as mg

class MinesweeperBotSettings():
    ''' Various data, needed to read the field information from screenshot.
    Different for different minesweeper version (Minesweeper X,
    Online minesweeper, Google minesweeper etc.
    Default one is for Minesweeper X.
    '''

    def __init__(self, field_color, cells_colors, minimum_cell_size=10):

        # Color used to find a grid. This should be the most central color
        # of a closed cell (or several colors if it is a chess-board-like,
        # as Google Minesweeper is)
        self.field_color = field_color

        # Distinct color of a cell (numbers + covered cells)
        # Whatever is not here will be considered zero.
        # Flagged mines will be stored in the Bot,
        # as they are easily confused with 3s
        self.cells_colors = cells_colors

        # Minimum size to be considered a potential cell (to rule out random small specks)
        self.minimum_cell_size = minimum_cell_size

    def copy(self):
        '''Create a copy of itself (to reuse default settings)
        '''
        new_settings = MinesweeperBotSettings(
            self.field_color.copy(),
            self.cells_colors.copy(),
            self.minimum_cell_size
        )
        return new_settings

SETTINGS_DEFAULT = MinesweeperBotSettings(
    field_color = [(192, 192, 192)],
    cells_colors = {
        (0, 0, 255): 1,
        (0, 128, 0): 2,
        (255, 0, 0): 3,
        (0, 0, 128): 4,
        (128, 0, 0): 5,
        (0, 128, 128): 6,
        (0, 0, 0): 7,
        (128, 128, 128): 8,
        (158, 0, 0): mg.CELL_EXPLODED_MINE,
        (235, 235, 235): mg.CELL_COVERED,
        }
    )

SETTINGS_MINESWEEPER_X = SETTINGS_DEFAULT.copy()

SETTINGS_MINESWEEPER_ONLINE = SETTINGS_DEFAULT.copy()
SETTINGS_MINESWEEPER_ONLINE.field_color = [(198, 198, 198)]

SETTING_GOOGLE_MINESWEEPER = SETTINGS_DEFAULT.copy()
SETTING_GOOGLE_MINESWEEPER.field_color = [(146, 217, 43), (155, 223, 54)]


class MinesweeperBot:
    ''' Class to play minesweeper from pixels: find the game on the screen,
    read the cells' values, click and so on
    '''

    def __init__(self, settings=SETTINGS_MINESWEEPER_X):
        # Bot settings, which colors are used to find and read the field
        self.settings = settings
        # The shape of the field (width and height for 2D games,
        # or higher order tuple for n-dimensional games)
        self.game_shape = None

        # Number of mines in a game (Tries to guess, if it is one of
        # a standard 2D sizes, but otherwise has to be set up manually)
        self.game_mines = 0

        # COordinates of the game n the screen
        self.cells_coordinates = None

    @staticmethod
    def add_alpha_color(colors):
        ''' If color in colors presented with 3 values (RGB),
        add another color, in RGBA and A = 255
        '''
        # Same logic, but can be used for either list or dict
        if isinstance(colors, list):
            for color in colors:
                if len(color) == 3:
                    colors.append(tuple(list(color) + [255]))
        if isinstance(colors, dict):
            new_colors = {}
            for color, value in colors.items():
                if len(color) == 3:
                    new_colors[tuple(list(color) + [255])] = value
            colors.update(new_colors)

    def find_game(self, image):
        '''Find game field by looking for squares of color "colors",
        placed in a grid. Return 2d array of (x1, y1, x2, y2) of found cells.
        image: PIL Image
        colors: list [color1, color2, ...] - any color will be matched
        '''

        def find_square(left, top):
            ''' Check if x, y is a left top corner of a rectangle
            pixels are from parent method.
            '''
            # Square should be the same color as it's top left corner
            color = pixels[left, top]

            # Find width
            right = left
            while right < image.size[0] and pixels[right+1, top] == color:
                right += 1

            # Find height
            bottom = top
            while bottom < image.size[1] and pixels[left, bottom + 1] == color:
                bottom += 1

            # Check if all the pixels are of the needed color
            for i in range(left, right + 1):
                for j in range(top, bottom + 1):

                    # This is not a one-color square
                    if pixels[i, j] != color:
                        return False, False, False, False

            return left, top, right, bottom

        def find_all_squares():
            ''' Find all squares of any of "colors" color.
            Return their 4-coordinates as a list
            '''

            # Will need to draw some pixel over
            draw = ImageDraw.Draw(image)

            # Scan image pixel by pixel, find one from "colors"
            for i in range(image.size[0]):
                for j in range(image.size[1]):
                    if pixels[i, j] in self.settings.field_color:

                        # When  found check if it is a square
                        # (technically, rectangles are okay too)
                        left, top, right, bottom = find_square(i, j)

                        # if it is - store 4 coordinates in "found"
                        if left and right - left > self.settings.minimum_cell_size:
                            found.append((left, top, right, bottom))

                            # Fill it with black so it would not be found again
                            draw.rectangle((left, top, right, bottom),
                                           fill="black")
                        else:
                            # Paint it over, so we will not have to test
                            # these pixels again
                            draw.line((left, top, right, top), fill="black")
                            draw.line((left, top, left, bottom), fill="black")

            return found

        def filter_grid(found):
            ''' Given found squares, only keep those that are on a "grid":
            their coordinates are the most repeating coordinates int the list
            '''
            # Count all x and y coordinates of all the squares we found
            x_count, y_count = {}, {}
            for left, top, right, bottom in found:
                x_count[left] = x_count.get(left, 0) + 1
                y_count[top] = y_count.get(top, 0) + 1
                x_count[right] = x_count.get(right, 0) + 1
                y_count[bottom] = y_count.get(bottom, 0) + 1

            # Calculate "weight" - how often this squares coordinates
            # are present in other squares
            found_with_weights = {}
            all_weights = []
            for left, top, right, bottom in found:
                weight = x_count[left] + y_count[top] + \
                        x_count[right] + y_count[bottom]
                found_with_weights[(left, top, right, bottom)] = weight
                all_weights.append(weight)

            # Find median of all weights. Anything higher or equal to than
            # will be in the final grid
            all_weights.sort()
            threshold = all_weights[len(all_weights) // 2]

            new_found = [coordinates
                         for coordinates, weight in found_with_weights.items()
                         if weight >= threshold]

            return new_found

        def deduce_game_parameters(found):
            '''From the found squares, deduce game dimensions and the number of mines
            '''

            game_width = len(set((left for left, _, _, _ in found)))
            game_height = len(set((top for _, top, _, _ in found)))
            game_mines = 0

            # Mine counts to recognize
            game_presets = {(8, 8): 10, (9, 9): 10, (16, 16): 40, (30, 16): 99,
                            (10, 8): 10, (18, 14): 40, (24, 20): 99}

            if (game_width, game_height) in game_presets:
                game_mines = game_presets[(game_width, game_height)]

            return (game_width, game_height), game_mines

        def arrange_cells(found):
            '''Arrange all found cells into a grid, in a form of NumPy array
            '''
            grid = np.array(found, dtype=object)
            grid = np.reshape(grid, list(self.game_shape) + [4])
            return grid

        # Add RGBA to acceptable colors (some screenshots have them)
        self.add_alpha_color(self.settings.field_color)

        # Pixels of the input image
        pixels = image.load()

        # We'll be putting found squares here:
        found = []

        # Find all potential squares
        found = find_all_squares()

        if len(found) < 10:
            print("Cannot fine the game")
            return

        # Filter those that are on the same grid
        found = filter_grid(found)

        # Determine game parameters (size, mines), from the found grid
        self.game_shape, self.game_mines = deduce_game_parameters(found)
        print(f"Found game of the size {self.game_shape}, " +
              f"assuming {self.game_mines} mines")

        # Sort them into rows and columns, store it in self.cells_coordinates
        self.cells_coordinates = arrange_cells(found)

    def read_field(self, image):
        ''' Read the information from the field: covered and uncovered cells,
        numbers, mines, etc. Return numpy array.
        '''

        def read_cell(image):
            ''' Read the data from the image of one cell
            '''
            pixels = image.load()

            # Go bottom to top (it helps tell apart 3 and a flag)
            for i in range(image.size[0]):
                for j in range(image.size[1]):
                    #print(pixels[middle, j])
                    if pixels[i, j] in self.settings.cells_colors:
                        return self.settings.cells_colors[pixels[i, j]]

            return 0

        # Add RGBA to acceptable colors (some screenshots have them)
        self.add_alpha_color(self.settings.cells_colors)

        # Create empty numpy array, and go through all cells, filling it
        field = np.zeros(self.game_shape, dtype=int)
        for i in range(self.game_shape[0]):
            for j in range(self.game_shape[1]):
                #if i != 5 or j != 6:
                #    continue
                left, top, right, bottom = self.cells_coordinates[i, j]
                # Add one pixel more, to be able to tell apart
                # covered and 0 (otherwise they are identical)
                cell_box = left - 1, top - 1, right + 1, bottom + 1
                cell = image.crop(cell_box)
                field[i, j] = read_cell(cell)

        return field

def main():
    ''' Some testing functions
    '''
    screenshot = Image.open("mso-1.png")
    bot = MinesweeperBot(SETTINGS_MINESWEEPER_ONLINE)
    bot.find_game(screenshot)
    screenshot = Image.open("mso-4.png")
    game = mg.MinesweeperGame()
    game.field = bot.read_field(screenshot)
    print(game.field2str(game.field))

if __name__ == "__main__":
    start = time.time()
    main()
    print(time.time() - start)
