''' Bot that plays minesweeper "from pixels", including finding the game
field on the screen, recognizing the numbers, clicking cells etc.
Uses minesweeper solver in minesweeper_solver.py
'''

# import pyautogui
import time
from PIL import Image, ImageDraw

MINIMUM_CELL_SIZE = 10


def find_game(image, colors):
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
                    if left and right - left > MINIMUM_CELL_SIZE:
                        found.append((left, top, right, bottom))

                        # Fill it with black so it would not be found again
                        draw.rectangle((left, top, right, bottom),
                                       fill=(0, 0, 0))
                    else:
                        # Paint it over, so we will not have to test
                        # these pixels again
                        draw.line((left, top, right, top), fill=(0, 0, 0))
                        draw.line((left, top, left, bottom), fill=(0, 0, 0))

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

        # Calculate "weight" - how often this squares coordinates are present
        # in other squares
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

    # Pixels of the input image
    pixels = image.load()

    # We'll be putting found squares here:
    found = []

    # Find all potential squares
    found = find_all_squares()

    # Filter those that are on the same grid
    found = filter_grid(found)

    # Sort them into rows and columns

    print(len(found))
    # print(found)


def main():
    ''' Some testing functions
    '''

    screenshot = Image.open("screenshot.png")
    colors = [(192, 192, 192)]
    find_game(screenshot, colors)


if __name__ == "__main__":
    start = time.time()
    for _ in range(10):
        main()
    print(time.time() - start)
