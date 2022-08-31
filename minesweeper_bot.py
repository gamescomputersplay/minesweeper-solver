''' Bot that plays minesweeper "from pixels", including finding the game
field on the screen, recognizing the numbers, clicking cells etc.
Uses minesweeper solver in minesweeper_solver.py
'''

#import pyautogui
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
                        draw.rectangle((left, top, right, bottom), fill=(0, 0, 0))

        return found

    # Pixels of the input image
    pixels = image.load()

    # We'll be putting found squares here:
    found = []

    # Find all potential squares
    found = find_all_squares()

    # Filter those that are on the same grid

    # Sort them into rows and columns

    print (found)

def main():
    ''' Some testing functions
    '''

    screenshot = Image.open("screenshot.png")
    colors = [(192, 192, 192)]
    find_game(screenshot, colors)


if  __name__ == "__main__":
    main()
