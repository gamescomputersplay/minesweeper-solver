# Minesweeper take 2

import numpy as np
import random
import time
import math


# Simulation settings

# Fields size and mines
#wid, hgt, mines = 8, 8, 10 # Beginner
#wid, hgt, mines = 16, 16, 40 # Intermediate
wid, hgt, mines = 30, 16, 99 # Expert
#wid, hgt, mines = 40, 24, 200 # Custom

#Simulation parameters
runs = 50000 # N of games to simulate
random.seed() # random seed
pre_generate_boards = False # to make sure it is the same boards between runs
report_interval = 5 # seconds, interval to show current stats



# Debug switches
show_moves = False # Show boards for each move
show_groups = False # Show generated groups and other solver data
show_method_stats = False # Show how many cells each algorythms found
show_game_results = False # Show result of each game

# array for pregeneratted boards
all_games = []


# Legend for the 'field' and 'opened' arrays
# 1-8: mines around
# 0: no mines around
# -1: mine (field)
# -2: closed cell (opened)
# -3: marked mine (opened)
# -4: exploded mine (opened)
# -5: incorrect mine (opened)
# -6: unknown, but not closed (for ne step analysis)



#################################################
# A few Helping Functions to go through the field

# returns list [(0,0), (0,1) .. (a-1,b-1)]
# kind of like "range" but for 2d array
def range2(a, b):
    permutations = []
    for j in range(b):
        for i in range(a):
            permutations.append((i,j))
    return permutations

# list of neighbouring cells' coordinates
# checking for not overstepping the border too 
def surroundings(x,y):
    surrounds = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i!=0 or j!=0) and x+i>=0 and y+j>=0 and x+i<wid and y+j<hgt:
                surrounds.append((x+i,y+j))
    return surrounds
        
    

######################################################
# Generate a field (np array) with this size and mines
# mines marked as -1
def generate_field():
    field = np.full((wid,hgt), 0)
    # Set mines
    for i in range(min(mines, int(wid*hgt*.9))): #limit density at 90%
        mine_set = False
        # Randomly pick a place that has no mines
        while not mine_set:
            x, y = random.randint(0,wid-1), random.randint(0,hgt-1)
            if field[x,y] == 0:
                field[x,y] = -1
                mine_set = True
    # Add numbers
    generate_numbers(field)
    return field

# (Re)calculate numbers for the field
# I use separate function, cause it is used twice:
# for generating the field and if the first click is a mine
def generate_numbers(field):
    for (i,j) in range2(wid, hgt):
        if field[i,j] != -1:
            field[i,j] = 0
            for (k,l) in surroundings(i,j):
                if field[k,l] == -1:
                    field[i,j] += 1

# Pre-generate all boards
def generate_all_games(runs):
    print ("Generating boards (", runs ,"): ", end="")
    for i in range(runs):
        all_games.append(generate_field())
        if i in [(runs//10)*(n+1) for n in range(10)]:
            print ("*", end="")
    print ("")
    return


##########################################
# Print mine field data in a human-friendly way
def print_field(field):
    for j in range(len(field[0])):
        row = ""
        for i in range(len(field)):
            if field[i,j] == -1 or field[i,j] == -3:
                row += "* "
            elif field[i,j] == -2:
                row += ". "
            elif field[i,j] == -4:
                row += "! "
            elif field[i,j] == -5:
                row += "X "
            else:   
                row += str(field[i,j]) + " "
        print (row)

###############################################################
# Some extra functions to help with simulating playing the game

# Are all the cells closed?
def has_all_closed(field):
    for (i,j) in range2(wid, hgt):
        if field[i,j] != -2:
            return False
    return True

# Are all the cells closed?
def has_closed(field):
    for (i,j) in range2(wid, hgt):
        if field[i,j] == -2:
            return True
    return False

# How many mines are there?
def count_mines(field):
    count=0
    for (i,j) in range2(wid, hgt):
        if field[i,j] == -1 or field[i,j] == -3:
            count += 1
    return count

# is the puzzle solved?
def is_solved(field, opened):
    for (i,j) in range2(wid, hgt):
        # Not solved if:
        # Marked mine that is not there
        # Has closed that is not a mine
        if    (opened[i,j]==-3 and field[i,j]!=-1) \
            or (opened[i,j]==-2 and field[i,j]>=0):
               return False
    return True
            
def show_incorrect_mines(field, opened):  
    for (i,j) in range2(wid, hgt):
        if field[i,j] != -1 and opened [i,j]==-3:
            opened [i,j]=-5


###################################
# Make one move: mark mines, safes
# In: game field, opened so far, list of mines/safe to click
# Out: new_open_field, mines_left, status
# Status: 0-keep_going, 1-won, 2-lost
def do_move(field, opened, mark_safe, mark_mines):
    status = 0
    
    #move mine for the very first mine click
    if has_all_closed(opened) and len(mark_safe)>0 and field[mark_safe[0]]==-1:
        rx, ry = random.randint(0,wid-1),random.randint(0,hgt-1)  
        while field[rx,ry]==-1:
            rx, ry = random.randint(0,wid-1),random.randint(0,hgt-1)
        field[rx,ry] = -1
        field[mark_safe[0]]= 0
        generate_numbers(field)

    # If there are zeros to open - put surroundings as extra safe cells
    keep_going = True
    while keep_going:
        keep_going = False
        to_add = []
        for cell in mark_safe:
            if field[cell] == 0:
                for add_cell in surroundings(cell[0], cell[1]):
                    to_add.append(add_cell)
        for cell in to_add:
            if cell not in mark_safe:
                mark_safe.append(cell)
                keep_going = True
            
    # Open all the safe cells
    for cell in mark_safe:
        if field[cell] == -1: # Blown up
            status = 2
            opened[cell] = -4 # Show the mine that killed player
            # Show incorrect mines
            show_incorrect_mines(field, opened)
        elif opened[cell] == -2: #normal open
            opened[cell] = field[cell]

    # Mark all the mines safe cells
    for cell in mark_mines:
        if opened[cell]==-2:
            opened[cell] = -3

    # No closed, yet hasn't won (that is, stuck) => Lost
    if not has_closed(opened) and not is_solved(field, opened):
        status = 2
        show_incorrect_mines(field, opened)

    # Won
    if status != 2 and is_solved(field, opened):
        status = 1 # won
        
    return (mines-count_mines(opened), status)


##########################
# Run one minesweeper game
# Out: result (1-won, 2-lost)
def minesweeper_game():
    if pre_generate_boards:
        field = all_games.pop()
    else:
        field = generate_field()
    opened = np.full((wid,hgt), -2)
    status = 0
    mines_left = mines
    if show_moves:
        print ("Generated field:")
        print_field(field)
        
    step=0
    while status==0:
        step+=1
        mark_safe, mark_mines = sweep(opened, mines_left)
        mines_left, status = do_move(field, opened, mark_safe, mark_mines)
        if show_moves:
            print ("Resulting field:")
            print_field(opened)
            print ("Status:", status)
    return status

                 
###########################
# Main simulation function:
# Run the game "runs" times
def simulation():

    # Margin of error for w wins in n simulations (95% confidence)
    def moe(won, n):
        w=won/n
        z=1.95 # for 95% confidence
        return z * math.sqrt(w*(1-w)/n) * 100
    
    if pre_generate_boards:
        generate_all_games(runs)

    t = time.time()
    
    last_report = time.time()
    won = 0
    out_format = "Games: {:d}, Won: {:d}, Success: {:4.2f}±{:4.2f}%, Speed: {:4.2f} games/s"
    out_format = "G {:d}, W {:d}, S {:4.2f}±{:4.2f}%, S {:4.2f} g/s"
    for i in range(runs):
        result = minesweeper_game()
        if show_game_results:
            print ("Game:", i)
        if result==1:
            won += 1
            if show_game_results:
                print ("Won")
        else:
            if show_game_results:
                print ("Lost")
        if (time.time()-last_report>report_interval):

            print (out_format.format(i+1, won, won*100/(i+1), moe(won, (i+1)), (i+1)/(time.time()-t)))
            last_report = time.time()
    print ("="*40)
    print (out_format.format(runs, won, won*100/runs,  moe(won, runs), runs/(time.time()-t)))
    print ("Time:", time.time()-t)
    



########################################################
# Main Solver Function: Get next moves for the given field
# In: Current opened field; N of mines, no_random - dont use guessing
# Out: ([List of safe], [List of mines])
def sweep(field, mines_left, do_random=True):
    
    #some initial setups and calculations
    mark_safe, mark_mines, b_safe, b_mines, r_safe, r_mines = [], [], [], [], [], []
    g_safe, g_mines, s_safe, s_mines, c_safe, c_mines = [], [], [], [], [], []
    p_safe, p_mines = [], []
    
    closed = get_all_closed(field)
    groups = get_groups(field)
    
    if show_groups:
        print ("Closed (", len(closed), "):", closed)
        print ("Groups (", len(groups), "):", groups)
        print ("Mines:", mines_left)
    
    # Various algorythms to determine safes and mines
    #------------------------------------------------
    # 1. Basic (compare N in the cell and N of surrounding cells)
    b_safe, b_mines = basic_sweep(groups)

    # 2. Group (compare groups)
    g_safe, g_mines = group_sweep(groups)

    # 3. Subgroup (generate and compare subgroups)
    if  len(b_safe)+len(b_mines) \
       +len(g_safe)+len(g_mines) == 0:
        s_safe, s_mines = subgroup_sweep(groups)
        pass

    
    
    # If 1,2,3 didn't bring anything
    if  len(b_safe)+len(b_mines) \
       +len(g_safe)+len(g_mines) \
       +len(s_safe)+len(s_mines) == 0:

        #5. CSP logic
        p_safe, p_mines = csp_sweep(groups, mines_left)
        
        # Generate a few things we'll use in methods 4 and in random
        coverage, covered_mines = get_best_coverage(groups)

        #4. Counter
        c_safe, c_mines = counter_sweep(closed, groups, mines_left, coverage, covered_mines)
        

    # if 1,2,3,4,5 didn't bring anything: Random
    if  len(b_safe)+len(b_mines) \
       +len(g_safe)+len(g_mines) \
       +len(s_safe)+len(s_mines) \
       +len(c_safe)+len(c_mines) \
       +len(p_safe)+len(p_mines)== 0 and do_random:


        r_safe, r_mines = random_sweep(closed, groups, mines_left,\
                                       coverage, covered_mines, field)


    # Out stats:
    if show_method_stats:
        print ("="*40)
        print ("Basic safe (", len(b_safe), "):", b_safe)
        print ("Basic mines (", len(b_mines), "):", b_mines)
        
        print ("Group safe (", len(g_safe), "):", g_safe)
        print ("Group mines (", len(g_mines), "):", g_mines)
        
        print ("Subgroup safe (", len(s_safe), "):", s_safe)
        print ("Subgroup mines (", len(s_mines), "):", s_mines)
        
        print ("Counter safe (", len(c_safe), "):", c_safe)
        print ("Counter mines (", len(c_mines), "):", c_mines)
        
        print ("CSP safe (", len(p_safe), "):", p_safe)
        print ("CSP mines (", len(p_mines), "):", p_mines)
        
        print ("Random (", len(r_safe), "):", r_safe)
        
    # Put all safe and mines together and return
    mark_safe = list(set(b_safe + g_safe + s_safe + c_safe + p_safe + r_safe))
    mark_mines = list(set(b_mines + g_mines + s_mines + c_mines + p_mines))
    return (mark_safe, mark_mines)



################################
# Some helpful solver's functions

# Get the list of all empty cells
def get_all_closed(field):
    closed = []
    for (i,j) in range2(wid, hgt):
        if field[i,j] == -2:
            closed.append((i,j))
    return set(closed)

    
# Generate list of GROUPS: [([list of cells], mines), ... ]
def get_groups(field):
    groups = []
    for (i,j) in range2(wid, hgt):
        if 0 < field[i,j] < 9:
            potential, closed, marked = 0, 0, 0
            group = []
            for cell in surroundings(i,j):
                if field[cell] == -2:
                    closed += 1
                    group.append(cell)
                if field[cell] == -3:
                    marked += 1
            if closed>0 and (set(group), field[i,j]-marked) not in groups:
                groups.append((set(group), field[i,j]-marked))
    return groups




######################################
# Various sweepeing algorythms

###########################
# The basic sweeping logic:
# - if there are 0 mines in group - all are safe
# - if there as many cells as mines - all are mines
def basic_sweep(groups):
    mark_safe = []
    mark_mines = []
    for group in groups:
        if group[1] == 0: # all safe
            for cell in group[0]:
                if cell not in mark_safe:
                    mark_safe.append(cell)
        if group[1] == len(group[0]): # all mines
            for cell in group[0]:
                if cell not in mark_mines:
                    mark_mines.append(cell)                
    return mark_safe, mark_mines


################################################################
# Cross checking mine groups, used in groups and subgroups sweep
# group1, group2 - groups to cross reference (1 should be a subgroup of 2)
# track extras - if you need to get the difference saved as new groups
# check_safe, check_mines - if you need only one type of search
#   (needed for subgroups)
def cross_check_groups(groups1, groups2, track_extras, mark_safe, mark_mines, \
                       check_safe=True, check_mines=True):
    extra_groups = []
    for group1 in groups1:
        for group2 in groups2:
            # 1 is subset of 2, 
            if group1 != group2 and group1[0].issubset(group2[0]):
                # they have same mines, then 2-1 is safe
                if group1[1]==group2[1] and check_safe:
                    for cell in group2[0].difference(group1[0]):
                        if cell not in mark_safe:
                            mark_safe.append(cell)
                # diff in mines == dif in size => 2-1 are mines
                elif group2[1]-group1[1]==len(group2[0])-len(group1[0]) \
                     and check_mines:
                    for cell in group2[0].difference(group1[0]):
                        if cell not in mark_mines:
                            mark_mines.append(cell)
                elif track_extras:
                    extra_groups.append((group2[0].difference(group1[0]), group2[1]-group1[1]))
    if track_extras:
        return extra_groups



###########################
# The group sweeping logic:
# cross-check all groups and see if there are mines/safes
def group_sweep(groups):
    mark_safe = []
    mark_mines = []
    extra_groups = []
    extra_groups = cross_check_groups( groups, groups, True, mark_safe, mark_mines)                
    cross_check_groups( extra_groups, groups, False, mark_safe, mark_mines)
    return mark_safe, mark_mines



##############################
# The sub-group sweeping logic:
# break dow all groups and compare it with groups
def subgroup_sweep(groups):
    mark_safe = []
    mark_mines = []

    # Generate "No more than N mines" subgroups
    def get_no_more_subgroups (group):
        nonlocal groups
        cells, mines = group[0], group[1]
        if len(cells)>=mines+1: # do i need +1 here?
            if group not in no_more_mines and group not in groups:
                no_more_mines.append(group)
            for cell in cells:
                get_no_more_subgroups ((cells.difference({cell}), mines))
        return

    # Generate "At least N mines" subgroups
    def get_at_least_subgroups (group):
        nonlocal groups
        cells, mines = group[0], group[1]
        if mines>=1:
            if group not in at_least_mines and group not in groups:
                at_least_mines.append(group)
            for cell in cells:
                get_at_least_subgroups ((cells.difference({cell}), mines-1))
        return
    
    # has at least N mines
    at_least_mines = []
    # has no more then N mines
    no_more_mines = []

    # Generate subgroups
    for group in groups:
        if len(group[0])<8:
            # <8, because takes too much time for single number
            # out in the open with no real effect
            get_at_least_subgroups(group)
            get_no_more_subgroups (group)
    # Use "At least" group just for safes
    cross_check_groups( at_least_mines, groups, False, mark_safe, mark_mines,\
                        True, False )
    # Use "No more" group just for mines
    cross_check_groups( no_more_mines, groups, False, mark_safe, mark_mines,\
                        False, True )

    return mark_safe, mark_mines



#################################################
# Get "Coverage" (way to groups co "cover" board
# to accomodate the most mines (ideally all of them)
# Used in Counter sweep and in Random
def get_best_coverage(groups):
    best_coverage = set()
    best_mines = 0
    for i in range(len(groups)*25):
        # experiemntaly, raising this number to 25 has the biggest impact
        # higher than that - more calculation for not too many new mines
        curr_coverage = set()
        curr_mines = 0
        for group in groups:
            if len(group[0].intersection(curr_coverage))==0:
                curr_coverage = curr_coverage.union(group[0])
                curr_mines += group[1]
        if curr_mines > best_mines:
            best_mines = curr_mines
            best_coverage = curr_coverage.copy()
        random.shuffle(groups)
    return best_coverage, best_mines


##################################################
# "Mine Counter" logic
# Fit as many mines into know groups, and tell safes and mines from that
def counter_sweep(closed, groups, mines_left, coverage, covered_mines):
    mark_safe = []
    mark_mines = []
    #coverage, covered_mines = get_best_coverage(groups)

    # All mines are covered, remaining cells are safe
    if mines_left == covered_mines:
        mark_safe = list(closed.difference(coverage))
    # Not covered cells == Not covered mines => are mines
    if len(closed.difference(coverage)) == mines_left-covered_mines:
        mark_mines = list(closed.difference(coverage))
    return mark_safe, mark_mines
    

#######################################################
# Some functions for CSP sweep
# Break down groups into non-overlaping clusters, retturn list of clusters
def get_clusters(groups):
    clusters = []
    gr = groups.copy() # I need a copy I can destruct
    while len (gr) > 0:
        clusters.append(gr.pop()[0])
        keep_doing = True
        while keep_doing:
            keep_doing = False
            for i in range(len(gr)):
                if len(gr[i][0].intersection(clusters[-1]))!=0:
                    clusters[-1] = clusters[-1].union(gr.pop(i)[0])
                    keep_doing = True
                    break
    # Convert to lists (because order is important)
    cluster_list = []
    for cluster in clusters:
        cluster_list.append(list(cluster))
    
    return cluster_list

# Generatee "conditions for cells in cluster
# in the form of binary mask - first groups are smaller digits
def get_conditions(cluster, groups):
    conditions = []
    for group in groups:
        condition_mask = 0
        for cell in group[0]:
            if cell in cluster:
                condition_mask += 2**cluster.index(cell)
        if condition_mask != 0 and (condition_mask, group[1]) not in conditions:
            conditions.append ((condition_mask, group[1]))
    return conditions

# Bruteforce solutions for these conditions
# solution is binary, representing mines and safes
def get_solutions(conditions, length, mines_left):
    solutions = []
    for i in range(2**length):
        for condition in conditions:
            # check i for a condition 
            if bin(condition[0] & i).count('1') != condition[1]:
                break
        else: # passed all conditions
            if bin(i).count('1') <= mines_left:
                solutions.append(i)
    return solutions

##########################################################
# CSP sweep logic
# Trest mine condditions as a CSP problem, bruteeforce the solutions
def csp_sweep(groups, mines_left):
    mark_safe = []
    mark_mines = []

    clusters = get_clusters(groups)
    for cluster in clusters:
        if len(cluster)<21:
            conditions = get_conditions(cluster, groups)
            solutions = get_solutions(conditions, len(cluster), mines_left)

            # look for "always mines" and "always safe" in solutions
            safe_mask = 0
            mine_mask = 2**len(cluster)-1
            # by running two bitmasks through all solutions
            for solution in solutions:
                #print (format(solution, '#016b'))
                safe_mask |= solution
                mine_mask &= solution
                
            # and then checking if there are 0s in safe mask and 1s in mine mask
            for i in range(2, len(cluster)+2): # 2, because string representation starts with "0b"
                #print (format(safe_mask, '#0'+str(len(cluster)+2)+'b')[i])
                if format(safe_mask, '#0'+str(len(cluster)+2)+'b')[i]=="0":
                    mark_safe.append(cluster[-i+1])
                if format(mine_mask, '#0'+str(len(cluster)+2)+'b')[i]=="1":
                    mark_mines.append(cluster[-i+1])
                    
    return mark_safe, mark_mines

    


#############################################################
# VERY naive random guessing - just random cell of all closed
def naive_random(cells):
    if len(cells)>0:
        rcell = cells[random.randint(0, len(cells)-1)]
        return [(rcell[0], rcell[1]),]
    else:
        return []



#################################################
# Fill the table with simple probability numbers
# (Highest probability from all groups)
def mine_probability(groups):
    mine_chance = np.full((wid,hgt), -1.0)
    for group in groups:
        chance = group[1]/len(group[0])
        for cell in group[0]:
            mine_chance[cell] = max(mine_chance[cell], chance)
    return mine_chance

#################################################
# Fill the table with opening probability
# (Not having miness in nall adjecent cells)
def opening_probability(mine_chance):
    opening_chance = np.full((wid,hgt), 0.0)
    for (i,j) in range2(wid, hgt):
        opening_chance[i,j] = 1
        for (k,l) in surroundings(i,j):
            if mine_chance[k,l]>0:
                opening_chance[i,j] *= (1-mine_chance[k,l])
    return opening_chance

# Factorial
def factorial(n):
    factorial=1
    for i in range(1,n+1):
        factorial *= i
    return factorial

# Combinations
def combinations (coll, items):
    return factorial(coll) / (factorial (items) * factorial (coll-items))


# Get sssoluttion with their weight, in the format [(soluttion, weight), ... ]
# wwhere weight - is the number of combinations of the remaining (not in this cluster) mines
def get_solutions_with_weights(conditions, length, closed, mines_left):
    # generate all permutations, check against conditions
    # find solutions that fit
    solutions = []
    total_weights = 0
    t=time.time()
    for i in range(2**length):
        for condition in conditions:
            if time.time()-t>10:
                print (conditions, length, closed, mines_left)
            # check i for condition 
            if bin(condition[0] & i).count('1') != condition[1]:
                break
        else: # passed all conditions
            if bin(i).count('1') <= mines_left:
                #print ("Solution:", format(i, '#016b'))
                remain_mines = mines_left-format(i, '#016b').count("1")
                remain_cells = len(closed)-length
                weight = combinations (remain_cells, remain_mines)
                solutions.append((i, weight))
                total_weights += weight
    return solutions, total_weights

# Get CSP chances from the solutions annnd weights
def get_csp_chances(solutions, cluster, total_weights):
    csp_chances = []
    length = len(cluster)
    cell_chance = [0] * length
    for i in range(2, length+2): # 2, because string representation starts with "0b"
        for solution in solutions:
            if format(solution[0], '#0'+str(length+2)+'b')[-i+1]=="1":
                cell_chance[i-2] += solution[1]/total_weights
                #print ("Cell", i-2, "Chance", solution[1]/total_weights)
    for i in range(length):
        csp_chances.append ((cluster[i], cell_chance[i]))
    return csp_chances


#############################################
# So, we have to guess after all...
# Picking the best random cell to click            
def random_sweep(closed, groups, mines_left, coverage, covered_mines, field):

    # Add group: all closed as one group
    #all_closed_group = []
    all_closed_group = [(closed, mines_left)]
    # seems to be worrking slightly better without it though
    

    # Add group: all closed "out there" minus minens we can cover
    not_covered_group = []
    if len(closed.difference(coverage))>0:
        not_covered_group = [(closed.difference(coverage), mines_left-covered_mines)]

    # Basic mine probability, 
    mine_chance = mine_probability( groups + all_closed_group + not_covered_group )

    # CSP Probability
    clusters = get_clusters(groups)
    csp_chances = []
    for cluster in clusters:
        if len(cluster)<21:
            conditions = get_conditions(cluster, groups)
            solutions, total_weights = \
                       get_solutions_with_weights(conditions, len(cluster), closed, mines_left)
            csp_chances += get_csp_chances(solutions, cluster, total_weights)
    for csp_chance in csp_chances:
        mine_chance[csp_chance[0]]=csp_chance[1]
        
    
    # Find the the lowest mine probabililty
    lowest_chance = 100
    lowest_cells = []
    for (i,j) in range2(wid, hgt):
        if mine_chance[i,j]!=-1.0 and mine_chance[i,j]==lowest_chance:
            lowest_cells.append((i,j))
        if mine_chance[i,j]!=-1.0 and mine_chance[i,j]<lowest_chance:
            lowest_cells = [(i,j),]
            lowest_chance = mine_chance[i,j]


    # Try removing the cell and see if solvable
    best_next_score = -1
    best_next_cells = []
    if len(lowest_cells)<10:
        for cell in lowest_cells:
            next_field=field.copy()
            next_field[cell]=-6
            next_safe, next_mine = sweep(next_field, mines_left, False)
            next_score = len(next_safe) * 2 + len(next_mine)
            if next_score == best_next_score:
                best_next_cells.append(cell)
            if next_score > best_next_score:
                best_next_cells = [cell,]
                best_next_score = next_score
        if best_next_cells != [] and best_next_score>0:
            lowest_cells = best_next_cells


    # Openning probability, find the highest       
    opening_chance = opening_probability(mine_chance)
    max_open_chance=0
    max_open_cells = []
    for cell in lowest_cells:
        if opening_chance[cell]==max_open_chance:
            max_open_cells.append(cell)
        if opening_chance[i,j]>max_open_chance:
            max_open_cells = [cell,]
            max_open_chance = opening_chance[i,j]


    # Random if several    
    mark_safe = naive_random(max_open_cells)
    
    return mark_safe, []

        
    

# run main program
simulation()
