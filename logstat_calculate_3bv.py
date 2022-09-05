'''Run some stats on the log file, namely 3BV
'''

INPUT_FILE = "log_control.log"

TARGET_WIDTH = 30
TARGET_HEIGHT = 16


def generate_neighbors(width, height):
    ''' Generate a list of neighbors for a MS field, represented as one line
    '''
    all_neighbors = {}
    # GO through all cells
    for current in range(width * height):

        currents_neighbors = []
        # These are 8 neighbor cells and how they are related to the center
        # in terms of coordinates on the line
        for shift in [-width - 1, -width, -width + 1, -1,
                      1, width-1, width, width + 1]:

            # A few conditions to make sure it is within borders
            # And don't spills over on the previous or next line
            if current + shift < 0 or current + shift >= width * height:
                continue
            if current % width == 0 and \
               shift in (-width - 1, -1, width - 1):
                continue
            if (current + 1) % width == 0 and \
               shift in (-width + 1, 1, width + 1):
                continue

            # If all is okay: add to the list of neighbors
            currents_neighbors.append(current + shift)

        all_neighbors[current] = currents_neighbors

    return all_neighbors


NEIGHBORS = generate_neighbors(TARGET_WIDTH, TARGET_HEIGHT)


def flood_fill_zero(field, start):
    '''In the field, flood fill zeros with spaces, starting at "start"
    '''
    # Make a list, we will need to mutate it
    field_list = list(field)

    # This is is a list of zeroes we found so far
    zeros = [start]

    while zeros:

        zero = zeros.pop()

        for zero_neighbor in NEIGHBORS[zero]:

            # Found a new zero among neighbors
            if field_list[zero_neighbor] == "0":
                # Add it to the list to go through
                zeros.append(zero_neighbor)
            # Whatever the cell is, mark it processed
            # by replacing it with space
            field_list[zero_neighbor] = " "

    # Construct the field back and return
    return "".join(field_list)


def get_3bv(field):
    ''' Calculate 3BV value for a minesweeper field, represented as a line
    '''
    count_of_3bv = 0

    # First, find and pop all zeros
    for position, cell_value in enumerate(field):
        if cell_value == "0":
            field = flood_fill_zero(field, position)
            count_of_3bv += 1

    # Then, go over the field again and count all numbers
    for cell_value in field:
        if cell_value not in (" ", "*"):
            count_of_3bv += 1

    return count_of_3bv


def recalculate_numbers(field):
    '''Recalculate all numbers in a field (since they may come from life
    games, it may be just mines surrounded by nothing.
    '''
    field_list = list(field)
    for position, cell_value in enumerate(field_list):
        # Do nothing with mines
        if cell_value == "*":
            continue
        # Count mines otherwise
        mines_count = 0
        for neighbor in NEIGHBORS[position]:
            if field_list[neighbor] == "*":
                mines_count += 1
        field_list[position] = str(mines_count)

    # Construct the field back and return
    return "".join(field_list)


def transpose_line(line):
    '''Log has data as "column-by-column", but we need it as "line-after-line",
    so we need to transpose the data in line, this is what this function does.
    '''
    # field[y, x]
    field = [
        ["" for _1 in range(TARGET_WIDTH)]
        for _2 in range(TARGET_HEIGHT)
    ]
    for position, current in enumerate(line):
        column = position // TARGET_HEIGHT
        row = position - column * TARGET_HEIGHT
        field[row][column] = current

    return "".join(["".join(line) for line in field])


def main():
    '''Run the test'''

    # Count all games
    game_count = 0
    # dict of 3BV stats: {3bv_value: games_with_this_3bv, ...}
    stats_3bv = {}

    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        for line in input_file:

            # Remove the line break in the end
            line = line.strip()

            # Skip if wrong game size (difficulty level)
            if len(line) != TARGET_WIDTH * TARGET_HEIGHT:
                continue

            game_count += 1

            # Make sure all mines, and only mines, are marked with *
            line = line.replace("!", "*").replace("X", " ")

            # line = transpose_line(line)

            # Recalculate numbers
            line = recalculate_numbers(line)

            # Calculate and store 3BV values
            this_3bv = get_3bv(line)
            stats_3bv[this_3bv] = stats_3bv.get(this_3bv, 0) + 1

        print(f"file: {INPUT_FILE}")
        print(f"Total: {game_count}")

        total_3bv = sum(value * count for value, count in stats_3bv.items())
        print(f"Average 3BV: {total_3bv / game_count}")

        all_3bv = []
        for value, count in stats_3bv.items():
            all_3bv.extend([value] * count)
        all_3bv.sort()
        print(f"Median 3BV: {all_3bv[len(all_3bv) // 2]}")


if __name__ == "__main__":
    main()
