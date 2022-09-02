''' Bot that plays minesweeper "from pixels", including finding the game
field on the screen, recognizing the numbers, clicking cells etc.
Uses minesweeper solver in minesweeper_solver.py
'''

import time

# import pyautogui
from PIL import Image, ImageDraw
import numpy as np

FIELD_COLORS_MINESWEEPER_X = [(192, 192, 192)]
FIELD_COLORS_MINESWEEPER_ONLINE = [(198, 198, 198)]
FIELD_COLORS_GOOGLE_MINESWEEPER = [(146, 217, 43), (155, 223, 54)]


class MinesweeperBot:
    ''' Class to play minesweeper from pixels: find the game on the screen,
    read the cells' values, click and so on
    '''

    MINIMUM_CELL_SIZE = 10

    def __init__(self):
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
        for color in colors:
            if len(color) == 3:
                colors.append(tuple(list(color) + [255]))

    def find_game(self, image, colors):
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
                    if pixels[i, j] in colors:

                        # When  found check if it is a square
                        # (technically, rectangles are okay too)
                        left, top, right, bottom = find_square(i, j)

                        # if it is - store 4 coordinates in "found"
                        if left and right - left > self.MINIMUM_CELL_SIZE:
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
        self.add_alpha_color(colors)

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


def main():
    ''' Some testing functions
    '''

    screenshot = Image.open("msx-beg.png")
    bot = MinesweeperBot()
    bot.find_game(screenshot, FIELD_COLORS_MINESWEEPER_X)

    screenshot = Image.open("mso-exp.png")
    bot = MinesweeperBot()
    bot.find_game(screenshot, FIELD_COLORS_MINESWEEPER_ONLINE)

    screenshot = Image.open("msg-int.png")
    bot = MinesweeperBot()
    bot.find_game(screenshot, FIELD_COLORS_GOOGLE_MINESWEEPER)

    screenshot = Image.open("mso-field.png")
    bot = MinesweeperBot()
    bot.find_game(screenshot, FIELD_COLORS_MINESWEEPER_ONLINE)

if __name__ == "__main__":
    start = time.time()
    main()
    print(time.time() - start)
