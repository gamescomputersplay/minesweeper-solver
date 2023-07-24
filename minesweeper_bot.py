''' Bot that plays minesweeper "from pixels", including finding the game
field on the screen, recognizing the numbers, clicking cells etc.
Uses minesweeper solver in minesweeper_solver.py
'''

import time
import math

import pyautogui
import keyboard
import numpy as np
from PIL import Image, ImageDraw

import minesweeper_game as mg
import minesweeper_solver as ms
import minesweeper_sim


class MinesweeperBotSettings():
    ''' Various data, needed to read the field information from screenshot.
    Different for different minesweeper version (Minesweeper X,
    Online minesweeper, Google minesweeper etc.
    Default one is for Minesweeper X.
    '''

    def __init__(self, field_color, samples_files,
                 cell_padding=1, click_pause=0.05, sample_sensitivity=3000):

        # Color used to find a grid. This should be the most central color
        # of a closed cell (or several colors if it is a chess-board-like,
        # as Google Minesweeper is)
        self.field_color = field_color

        # Load sample pictures of cells
        self.samples = [(Image.open(file), value)
                        for file, value in samples_files.items()]

        # How many pixels to pad when cut out a cell picture
        self.sample_sensitivity = sample_sensitivity

        # Minimum size to be considered a potential cell
        # (to rule out random small specks)
        self.minimum_cell_size = 10

        # How many pixels to pad when cut out a cell picture
        self.cell_padding = cell_padding

        # Pause after a click (to give game  time to react)
        self.click_pause = click_pause


# Settings for classic minesweeper versions
# (2000s, XPs, MInesweeper X, Vienna, Arbiter)
# Note, it is calibrated for 100% screen scale, will not work at 125% or others
SETTINGS_MINESWEEPER_CLASSIC = MinesweeperBotSettings(
    field_color=[(192, 192, 192), (192, 192, 192, 255)],
    samples_files={
        "./samples/msx-0.png": 0,
        "./samples/msx-1.png": 1,
        "./samples/msx-2.png": 2,
        "./samples/msx-3.png": 3,
        "./samples/msx-4.png": 4,
        "./samples/msx-5.png": 5,
        "./samples/msx-6.png": 6,
        "./samples/msx-7.png": 7,
        "./samples/msx-8.png": 8,
        "./samples/msx-mine.png": mg.CELL_MINE,
        "./samples/msx-flag.png": mg.CELL_MINE,
        "./samples/msx-covered.png": mg.CELL_COVERED,
        "./samples/msx-explosion.png": mg.CELL_EXPLODED_MINE,
        }
    )

SETTINGS_MINESWEEPER_4D = MinesweeperBotSettings(
    field_color=[(153, 153, 153), (153, 153, 153, 255)],
    samples_files={
        "./samples/4d-0-a.png": 0,
        "./samples/4d-0-b.png": 0,
        "./samples/4d-1-a.png": 1,
        "./samples/4d-1-b.png": 1,
        "./samples/4d-2-a.png": 2,
        "./samples/4d-2-b.png": 2,
        "./samples/4d-3-a.png": 3,
        "./samples/4d-3-b.png": 3,
        "./samples/4d-4-a.png": 4,
        "./samples/4d-4-b.png": 4,
        "./samples/4d-5-a.png": 5,
        "./samples/4d-5-b.png": 5,
        "./samples/4d-6-a.png": 6,
        "./samples/4d-7-a.png": 7,
        "./samples/4d-7-b.png": 7,
        "./samples/4d-8-a.png": 8,
        "./samples/4d-9-a.png": 9,
        "./samples/4d-10-a.png": 10,
        "./samples/4d-11-a.png": 11,

        "./samples/4d-flag-a.png": mg.CELL_MINE,
        "./samples/4d-flag-b.png": mg.CELL_MINE,
        "./samples/4d-mine-a.png": mg.CELL_MINE,
        "./samples/4d-false-a.png": mg.CELL_FALSE_MINE,
        "./samples/4d-covered.png": mg.CELL_COVERED,
        "./samples/4d-explosion.png": mg.CELL_EXPLODED_MINE,
        },
    cell_padding=-2,
    click_pause=0.5,
    sample_sensitivity=10000
    )


# Should the bot stop when cell is not recognized
# False: No, just replace it with 0s. This will work if you are playing 1 game
# at a time and there is a popup message in the  end (this happens in
# 4D minesweeper, for example)
# True: Yes, throw an Exception, save the unknown cell as a new file in
# "samples". This might bee helpful when you "teach" program to read
# new interface
STOP_AT_UNKNOWN_CELL = False


class MinesweeperBot:
    ''' Class to play minesweeper from pixels: find the game on the screen,
    read the cells' values, click and so on
    '''

    def __init__(self, settings=SETTINGS_MINESWEEPER_CLASSIC,
                 mines=None, is_4d=False):
        ''' IN:
                - settings: a MinesweeperBotSettings with color settings
                    to read the field from the screenshot
                - mines: override a number of mines (if it is not a standard
                    field, or standard field with custom mine number)
                - is_4d: Flag for  Steam's 4D minesweeper. Would transform
                    2D field on the screen into a 4D field to play off of.
                    Then when it comes to clicks, transforms it back to  2D
        '''
        # Bot settings, which colors are used to find and read the field
        self.settings = settings

        # The shape of the field (width and height for 2D games,
        # or higher order tuple for n-dimensional games)
        self.game_shape = None

        # Number of mines in a game (Tries to guess, if it is one of
        # a standard 2D sizes, but otherwise has to be set up manually)
        self.game_mines = mines

        # Is it the 4D minesweeper version that you can find on Steam?
        self.is_4d = is_4d

        # Coordinates of the game on the screen
        self.cells_coordinates = None

        # Placeholder for the solver
        self.solver = None

        self.bot_stat = minesweeper_sim.SolverStat()

        # Cell recognition cache
        self.cell_cache = {}

        # Default pause between clicks is is 0.1 (meaning there will be
        # 40 seconds of pause on the Expert game). Let's speed it up.
        pyautogui.PAUSE = 0.01

    @staticmethod
    def transform_to_4d(field):
        ''' Transform 2D field (the way is it read from the picture)
        to the 4D field (to play it)
        '''
        # Only works for 4d cubes (all 4d sized should be equal)
        new_side = int(math.sqrt(field.shape[0]))
        new_shape = (new_side, new_side, new_side, new_side)
        field_4d = np.zeros(new_shape, dtype=int)

        # Go though all the current coordinates
        # and fill the new 4d field
        # order is: big rows, big columns, small rows, small columns
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):

                new_x = j // new_side
                new_y = i // new_side
                new_z = j % new_side
                new_w = i % new_side

                field_4d[new_x, new_y, new_z, new_w] = field[i, j]

        return field_4d

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
                        # and it is "square-ish" enough
                        # store 4 coordinates in "found"
                        if left and \
                           right - left > self.settings.minimum_cell_size and \
                           (bottom - top) > 0 and \
                           1.1 > (right-left) / (bottom - top) > 0.9:
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
            game_presets = {(8, 8): 10, (9, 9): 10, (16, 16): 40, (30, 16): 99}

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
            print("Cannot find the game")
            return False

        # Filter those that are on the same grid
        found = filter_grid(found)

        # Determine game parameters (size, mines), from the found grid
        self.game_shape, deduced_mines = deduce_game_parameters(found)
        print(f"Found game of the size {self.game_shape}")

        # If no mine count passed to the bot - try to assume from the game size
        if self.game_mines is None:
            self.game_mines = deduced_mines
            print(f"Assuming {self.game_mines} mines")
        else:
            print(f"Mines are set to {self.game_mines}")

        # Sort them into rows and columns, store it in self.cells_coordinates
        self.cells_coordinates = arrange_cells(found)

        # If it is a 4D game, override the settings we use to initiate solver
        # It only works for perfectly square games
        # (all 4 dimensions have to be  equal)
        if self.is_4d:
            field_side = int(math.sqrt(self.game_shape[0]))
            shape_4d = tuple(field_side for _ in range(4))
            settings = mg.GameSettings(shape_4d, self.game_mines)
        else:
            settings = mg.GameSettings(self.game_shape, self.game_mines)
        # Initiate solver
        self.solver = ms.MinesweeperSolver(settings)

        return True

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

        # Flag if we printed "unknown cells" warning
        # (so not to show it 100500 times)
        warning_was_fired = False

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
                # If STOP_AT_UNKNOWN_CELL is set
                # Save the sample, out a message
                if cell_value is None:
                    cell_value = 0
                    if STOP_AT_UNKNOWN_CELL:
                        filename = f"./samples/unknown-{i}-{j}.png"
                        cell.save(filename)
                        raise ValueError(
                            f"Can't read cell at ({i}, {j})," +
                            f"saved as {filename}")
                    if not warning_was_fired:
                        print("Some cells were not recognized, but ",
                              "STOP_AT_UNKNOWN_CELL is set to False. ")
                        warning_was_fired = True

                # Otherwise, store the read number in field array
                field[i, j] = cell_value

        return field

    def do_clicks(self, safe, mines):
        '''Given the safe and mines coordinates, do the clicks
        '''
        for button, coord_list in zip(("right", "left"), (mines, safe)):
            if not coord_list:
                continue
            for coord in coord_list:

                if self.is_4d:
                    x_4d, y_4d, z_4d, w_4d = coord
                    field_side = int(math.sqrt(self.game_shape[0]))

                    # This part is a mess, but whatever bugs there are, they
                    # seem to have cancelled each other out, so it works
                    x_2d = y_4d * field_side + w_4d
                    y_2d = x_4d * field_side + z_4d
                    left, top, right, bottom = \
                        self.cells_coordinates[x_2d, y_2d]

                else:
                    left, top, right, bottom = self.cells_coordinates[coord]

                # Actual clicking
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
        run a solver for the next move, click the cells
        '''

        def log_field(field, filename="log.log"):
            '''Save the field into a log file. For debugging purposes
            '''
            game_settings = mg.GameSettings(self.game_shape, self.game_mines)
            game = mg.MinesweeperGame(game_settings)
            field_str = game.export_field(field)
            with open(filename, "a", encoding="utf-8") as logfile:
                logfile.write(f"{field_str}\n")

        actually_do_clicks = False
        # Not screenshot means this is not a test,
        # we are actually playing the game
        if screenshot is None:
            actually_do_clicks = True
            screenshot = pyautogui.screenshot()

        # Read the field
        field = self.read_field(screenshot)

        # Check if the game is over, obe way or another
        if self.is_dead(field):
            log_field(field)
            self.bot_stat.add_game(mg.STATUS_DEAD)
            return mg.STATUS_DEAD
        if not self.has_covered(field):
            log_field(field)
            self.bot_stat.add_game(mg.STATUS_WON)
            return mg.STATUS_WON

        # For 4D Game: do the transformation for the solver
        if self.is_4d:
            field = self.transform_to_4d(field)

        # Print out what we have read (for debugging)
        # game = mg.MinesweeperGame()
        # print(game.field2str(field))

        # Get the solution to the current field
        safe, mines = self.solver.solve(field)

        # Track statistics
        self.bot_stat.add_move(self.solver.last_move_info, safe, mines)

        # If it is not testing - do the clicks
        if actually_do_clicks:
            self.do_clicks(safe, mines)

            # Quick pause to make sure game have time to react
            # and we get updated info on the screen
            # Seems to be REALLY important to the win rate
            time.sleep(self.settings.click_pause)

        # This status is more for consistency
        return mg.STATUS_ALIVE


def use_bot(games_to_play=100, settings=SETTINGS_MINESWEEPER_CLASSIC,
            mines=None, is_4d=None):
    ''' Play several games. See MinesweeperBot class description for
    details about the parameters.
    '''

    # Create a new bot object
    bot = MinesweeperBot(settings=settings, mines=mines, is_4d=is_4d)

    # Find the game on the screen
    game_found = bot.find_game()

    if not game_found:
        return

    wins = 0

    for game in range(games_to_play):

        print(f"Game #{game + 1}.", end=" ")

        # Endless cycle to make moves
        while True:

            # Read a screen, do clicks
            result = bot.make_a_move()

            # Out if we won or lost
            if result == mg.STATUS_DEAD:
                print("I died.", end=" ")
                break
            if result == mg.STATUS_WON:
                print("I won.", end=" ")
                wins += 1
                break

        print(f"Win rate: {wins / (game + 1):.2%}")

        # Click the new game button (every time, except the last)
        if game < games_to_play - 1:
            left = bot.cells_coordinates[0, 0, 0]
            right = bot.cells_coordinates[-1, 0, 2]
            top = bot.cells_coordinates[0, 0, 1]
            new_game = ((left + right) // 2, top - 30)

            # This pause is for humans watching the game
            time.sleep(0.5)
            pyautogui.click(new_game)
            # This pause for minesweeper to refresh the screen
            time.sleep(0.3)

    print(bot.bot_stat)


def main():
    '''Run the bot program
    '''
    # Playing regular classic minesweeper
    use_bot(10)

    # Playing 4D Steam Minesweeper
    # use_bot(1, settings=SETTINGS_MINESWEEPER_4D, mines=20, is_4d=True)


if __name__ == "__main__":
    start = time.time()
    keyboard.add_hotkey('f10', main)
    keyboard.wait('esc')
    print(time.time() - start)
