''' Bot that plays minesweeper "from pixels", including finding the game
field on the screen, recognizing the numbers, clicking cells etc.
Uses minesweeper solver in minesweeper_solver.py
'''

import time

import pyautogui
import keyboard
import numpy as np
from PIL import Image, ImageDraw

import minesweeper_game as mg
import minesweeper_solver as ms


class MinesweeperBotSettings():
    ''' Various data, needed to read the field information from screenshot.
    Different for different minesweeper version (Minesweeper X,
    Online minesweeper, Google minesweeper etc.
    Default one is for Minesweeper X.
    '''

    def __init__(self, field_color, samples_files):

        # Color used to find a grid. This should be the most central color
        # of a closed cell (or several colors if it is a chess-board-like,
        # as Google Minesweeper is)
        self.field_color = field_color

        # Load sample pictures of cells
        self.samples = [(Image.open(file), value)
                        for file, value in samples_files.items()]

        # How many pixels to pad when cut out a cell picture
        self.sample_sensitivity = 3000

        # Minimum size to be considered a potential cell
        # (to rule out random small specks)
        self.minimum_cell_size = 10

        # How many pixels to pad when cut out a cell picture
        self.cell_padding = 1


SETTINGS_MINESWEEPER_X = MinesweeperBotSettings(
    field_color=[(192, 192, 192), (192, 192, 192, 255)],
    samples_files={
        "samples/msx-125-0.png": 0,
        "samples/msx-125-1.png": 1,
        "samples/msx-125-2.png": 2,
        "samples/msx-125-3.png": 3,
        "samples/msx-125-4.png": 4,
        "samples/msx-125-5.png": 5,
        "samples/msx-125-6.png": 6,
        "samples/msx-125-7.png": 7,
        "samples/msx-125-covered.png": mg.CELL_COVERED,
        "samples/msx-125-mine.png": mg.CELL_MINE,
        "samples/msx-125-flag.png": mg.CELL_MINE,
        "samples/msx-125-explosion.png": mg.CELL_EXPLODED_MINE,
        "samples/msx-125-false.png": mg.CELL_FALSE_MINE,
        }
    )

# SETTING_GOOGLE_MINESWEEPER.field_color = [(146, 217, 43), (155, 223, 54)]


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

        # Coordinates of the game on the screen
        self.cells_coordinates = None

        # Placeholder for the solver
        self.solver = None

        # Cell recognition cache
        self.cell_cache = {}

        # Default pause between clicks is is 0.1 (meaning there will be
        # 40 seconds of pause on the Expert game). Let's speed it up.
        pyautogui.PAUSE = 0.01

    def find_game(self, image=None):
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

                        # If the square is found and it is large enough,
                        # store 4 coordinates in "found"
                        if left and \
                           right - left > self.settings.minimum_cell_size:
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
            '''From the found squares, deduce game
            dimensions and the number of mines.
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

        # Take a screenshot, if needed
        if image is None:
            image = pyautogui.screenshot()

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

        # Initiate solver
        settings = mg.GameSettings(self.game_shape, self.game_mines)
        self.solver = ms.MinesweeperSolver(settings)

    def read_field(self, image):
        ''' Read the information from the field: covered and uncovered cells,
        numbers, mines, etc. Return numpy array.
        '''

        def get_difference(image1, image2):
            '''Calculate difference in pixel values between 2 images.
            '''
            pixels1 = image1.load()
            pixels2 = image2.load()
            difference = 0
            for i in range(min(image1.size[0], image2.size[0])):
                for j in range(min(image1.size[1], image2.size[1])):
                    for position in range(3):
                        difference += abs(pixels1[i, j][position] -
                                          pixels2[i, j][position])
            return difference

        def get_image_hash(image):
            ''' Calculate hash of otherwise unhashable image
            '''
            image_data = []
            pixels = image.load()
            for i in range(image.size[0]):
                for j in range(image.size[1]):
                    image_data.append(pixels[i, j])
            return hash(tuple(image_data))

        def read_cell(image):
            ''' Read the data from the image of one cell
            '''
            # Check if we maybe saw this one before
            image_hash = get_image_hash(image)
            if image_hash in self.cell_cache:
                return self.cell_cache[image_hash]

            # Compare the image with known  cell samples
            best_fit_difference = None
            best_fit_value = None
            for sample, value in self.settings.samples:
                # Calculate difference with a sample
                difference = get_difference(sample, image)
                # Check with all and use the closest one, but only i
                # f difference is smaller than sensitivity.
                if difference < self.settings.sample_sensitivity:
                    if best_fit_difference is None \
                       or difference < best_fit_difference:
                        best_fit_difference = difference
                        best_fit_value = value
            if best_fit_value is not None:
                # Store the result in cache
                self.cell_cache[image_hash] = best_fit_value
                return best_fit_value
            return None

        # Create empty numpy array, and go through all cells, filling it
        field = np.zeros(self.game_shape, dtype=int)
        for i in range(self.game_shape[0]):
            for j in range(self.game_shape[1]):

                left, top, right, bottom = self.cells_coordinates[i, j]
                # Add one pixel more, to be able to tell apart
                # covered and 0 (otherwise they are identical)
                cell_box = left - self.settings.cell_padding, \
                    top - self.settings.cell_padding, \
                    right + self.settings.cell_padding, \
                    bottom + self.settings.cell_padding
                cell = image.crop(cell_box)
                cell_value = read_cell(cell)
                # Cell not recognized (difference is higher than sensitivity)
                # Save the sample, out a message
                if cell_value is None:
                    filename = f"samples/unknown-{i}-{j}.png"
                    cell.save(filename)
                    raise Exception(
                        f"Can't read cell at ({i}, {j}), saved as {filename}")
                # Otherwise, store the read number in field array
                field[i, j] = cell_value

        return field

    def do_clicks(self, safe, mines):
        '''Given the safe and mines coordinates, do the clicks
        '''
        for button, coord_list in zip(("left", "right"), (safe, mines)):
            if not coord_list:
                continue
            for coord in coord_list:
                left, top, right, bottom = self.cells_coordinates[coord]
                x_coord = (left + right) // 2
                y_coord = (top + bottom) // 2
                pyautogui.click(x_coord, y_coord, button=button)

    def is_dead(self, field):
        '''Check if there is an exploded mine on the field,
        which means the game is over
        '''
        for i in range(self.game_shape[0]):
            for j in range(self.game_shape[1]):
                if field[i, j] == mg.CELL_EXPLODED_MINE:
                    return True
        return False

    def has_covered(self, field):
        '''Check if there are any covered cells left
        '''
        for i in range(self.game_shape[0]):
            for j in range(self.game_shape[1]):
                if field[i, j] == mg.CELL_COVERED:
                    return True
        return False

    def make_a_move(self, screenshot=None):
        ''' Read the situation on the board,
        run a solver for teh next move, click the cells
        '''

        actually_do_clicks = False
        # Not screenshot means this is not a test,
        # we are actually playing the game
        if screenshot is None:
            actually_do_clicks = True
            screenshot = pyautogui.screenshot()

        # Read the field
        field = self.read_field(screenshot)
        #game = mg.MinesweeperGame()
        #print(game.field2str(field))

        # Check if the game is over, obe way or another
        if self.is_dead(field):
            mg.log_field(field)
            return mg.STATUS_DEAD
        if not self.has_covered(field):
            mg.log_field(field)
            return mg.STATUS_WON

        # Get the solution to the current field
        safe, mines = self.solver.solve(field)

        # If it is not testing - do the clicks
        if actually_do_clicks:
            self.do_clicks(safe, mines)

        # This status is more for consistency
        return mg.STATUS_ALIVE



def use_bot(games_to_play=100):
    ''' Play several games
    '''

    # Create a new bot object
    bot = MinesweeperBot(SETTINGS_MINESWEEPER_X)

    # Find the game on the screen
    bot.find_game()

    wins = 0

    for game in range(games_to_play):

        print(f"Game #{game + 1}.", end=" ")

        # Endless cycle to make moves
        while True:

            # Read a screen, do clicks
            result = bot.make_a_move()

            # Quick pause to make sure game reacts
            time.sleep(.01)

            # Out if we won or lost
            if result == mg.STATUS_DEAD:
                print("I died.", end=" ")
                break
            if result == mg.STATUS_WON:
                print("I won.", end=" ")
                wins += 1
                break

        print(f"Win rate: {wins / (game + 1):.2%}")

        # CLick new game button
        left = bot.cells_coordinates[0, 0, 0]
        right = bot.cells_coordinates[-1, 0, 2]
        top = bot.cells_coordinates[0, 0, 1]
        new_game = ((left + right) // 2, top - 30)
        time.sleep(0.5)
        pyautogui.click(new_game)

def main():
    '''Run the bot program
    '''
    use_bot(10)

if __name__ == "__main__":
    start = time.time()
    keyboard.add_hotkey('f10', main)
    keyboard.wait('esc')
    print(time.time() - start)
