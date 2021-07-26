# Minesweeper solver
# Uses MinesweeperX__1.15
# https://youtu.be/ADKUTmDZtGY

from PIL import Image
import keyboard
import pyautogui
import time
import random
from pynput.mouse import Button, Controller
import cv2
import numpy as np
import matplotlib.pyplot as plt

# edit these to make sure it finds the game game on the screen
square_size = 20

#size of those gray borders (used for finding the game)
grey_border_left = 11
grey_border_right = 10
grey_border_top = 65
grey_border_bottom = 11

#online full hd
##grey_border_left = 9
##grey_border_right = 10
##grey_border_top = 62
##grey_border_bottom = 9

#office laptop
##square_size = 16
##grey_border_left = 8
##grey_border_right = 8
##grey_border_top = 50
##grey_border_bottom = 8

game_x = 0
game_y = 0
wid = 0
hgt = 0
field =[]
has_opening = False

#timers
t_screen = 0
t_read = 0
t_sweep = 0
t_click = 0

#sweep method stats
s_basic = 0
s_advance = 0
s_genious = 0
s_counter = 0
s_csp = 0
s_random = 0

#result stats
games_total = 0
games_won = 0
last_guessing = 0
success_rate = []

mouse = Controller()


color_mine_1 = (62,115,135)
color_safe_1 = (65,120,140)
color_mine_2 = (1,80,160)
color_safe_2 = (2,85,165)
color_mine_3 = (135,150,90)
color_safe_3 = (140,155,95)
color_counter = (12,230,41)
color_csp = (255,255,0)
color_first = (180,180,180)
color_random = (255,0,0)

# is the right shade of gray
def is_gray(px):
    if px==(189, 189, 189) or px==(189, 189, 189, 255) or px==(192, 192, 192) or px==(192, 192, 192, 255):
        return True
    return False

# is it *the* gray square
def the_grey_square(x,y):
    global px
    #first, find the left top corner gray
    for j in range(-6, 0):
        for i in range(-6, 0):
            if is_gray(px[x+i,y+j]):
                for j2 in range(6):
                    for i2 in range(6):
                        if not is_gray(px[x+i+i2,y+j+j2]):
                            return (0,0)
                return (x+i,y+j)
    return (0,0)

# find the left grey corner
def find_grey_square():
    global px
    global im
    step = 6
    for j in range (0, im.size[1], step):
        for i in range (0, im.size[0], step):
            if is_gray(px[i,j]):
                #print (i,j)
                test_gray=the_grey_square(i,j)
                if test_gray != (0,0) :
                    return test_gray
    return False

#calculate the cells from the grey square               
def find_grid (start):
    global px
    global game_x
    global game_y
    global wid
    global hgt
    global square_size
    global grey_border_left
    global grey_border_right
    global grey_border_top
    global grey_border_bottom	
    x0 = start[0]
    y0 = start[1]
    i=0
    while is_gray(px[x0+i,y0]):
        i += 1
    game_x = x0+grey_border_left
    wid=(i-grey_border_left-grey_border_right)//square_size
    i=0
    while is_gray(px[x0,y0+i]):
        i += 1
    game_y = y0+grey_border_top
    hgt=(i-grey_border_bottom-grey_border_top)//square_size
    #print ("Field corner: ", game_x, game_y)

#get a cell
def get_cell(x,y):
    global im
    global game_x
    global game_y
    global square_size
    box = (game_x+x*square_size, game_y+y*square_size, game_x+(x+1)*square_size, game_y+(y+1)*square_size)
    region = im.crop(box)
    return region

#average a cell's rgb
def ave_cell(cell):
    step=3
    p = cell.load()
    r,g,b = 0, 0, 0
    for i in range (cell.size[0]):
        for j in range (cell.size[1]):
            r += p[i,j][0]
            g += p[i,j][1]
            b += p[i,j][2]
    return (r,g,b)

#image recognize what's in the cell
# this is actually what needs to be changed to make it work with other minesweepers
# program will save unrecognized cell + RGB data, add the conditions here
def read_a_cell(cell):
    rgb = ave_cell(cell)
    if rgb == (60309, 60309, 76449) or (rgb[0] == rgb[1] and  rgb[0] > 60000 and rgb[0] <62500 and  rgb[2] > 76000 and rgb[2] <78500):
        return 1
    if rgb == (52611, 65402, 52611) or (rgb[0] == rgb[2] and  rgb[0] > 52000 and rgb[0] <55000 and  rgb[1] > 65000 and rgb[1] <68000):
        return 2
    if rgb == (78876, 53376, 53376) or (rgb[1] == rgb[2] and  rgb[0] > 78000 and rgb[0] <81000 and  rgb[1] > 53000 and rgb[1] <56000):
        return 3
    if rgb == (55587, 55587, 66444) or (rgb[0] == rgb[1] and  rgb[0] > 55000 and rgb[0] <58000 and  rgb[2] > 66000 and rgb[2] <69000):
        return 4
    if rgb == (64790, 50866, 50866) or (rgb[1] == rgb[2] and  rgb[0] > 64000 and rgb[0] <67500 and  rgb[1] > 50000 and rgb[1] <53500):
        return 5
    if rgb == (50423, 64637, 64637) or (rgb[1] == rgb[2] and  rgb[0] > 50000 and rgb[0] <52500 and  rgb[1] > 64000 and rgb[1] <67500):
        return 6
    if rgb == (58983, 58983, 58983) or (rgb[0] == rgb[1] and  rgb[0] == rgb[2] and rgb[0] > 58000 and rgb[0] <61000):
        return 7
    if rgb == (64260, 64260, 64260) or (rgb[0] == rgb[1] and  rgb[0] == rgb[2] and rgb[0] > 66000 and rgb[0] <67000):
        return 8
    if (rgb[1] == rgb[2] and rgb[0] > 63300 and rgb[0] <65300 and rgb[1] > 8200 and rgb[1] <10500) \
       or (rgb[0] == rgb[1] and  rgb[0] == rgb[2] and rgb[0] > 50000 and rgb[0] <52000):
        return -100 # an exploded mine (game over)
    if rgb == (72272, 72272, 72272) or (rgb[0] == rgb[1] and  rgb[0] == rgb[2] and rgb[0] > 72000 and rgb[0] <74500):
        return 0
    if rgb == (75599, 75599, 75599) or (rgb[0] == rgb[1] and  rgb[0] == rgb[2] and rgb[0] > 75000 and rgb[0] <77500) \
       or (rgb[0] - rgb[1]  < 600 and  rgb[1] == rgb[2] and rgb[0] > 75000 and rgb[0] <77500):
        return -1 # closed cell
    if rgb == (70702, 63861, 63861) or (rgb[1] == rgb[2] and  rgb[0] > 70000 and rgb[0] <72500 and  rgb[1] > 63000 and rgb[1] <66000):
        return -2 # marked mine
    print ("cant recognize: ", rgb)
    cell.save("cant-recognise.png")
    return -999 # not recognized

#read data from a pic to an array
def read_data():
##    global fields
##    fields = []
    field.clear()
    for i in range (wid):
        field.append([])
        for j in range (hgt):
            cell = get_cell(i,j)
            result = read_a_cell(cell)
            if result != None and result != -999 and result != -100:
                field[i].append(result)
            else:
                return result
    return

# print the field nicely (for debug)
def nice_print():
    for j in range (hgt):
        string = ""
        for i in range (wid):
            if field[i][j] == -1: 
                string += "0"
            elif field[i][j] == -2:
                string += "*"
            elif field[i][j] == 0:
                string += " "
            else: 
                string += str( field[i][j] )
        print (string)

def right_clicks (targets, color=(255, 255, 255)):
    delay = 0#.1
    global square_size
    global field
    global display
    color = (color[2], color[1], color[0])
    
    #print (targets)
    for i in range (len(targets)):
        mouse.position = (game_x+targets[i][0]*square_size+10, game_y+targets[i][1]*square_size+10)
        mouse.click(Button.right)
        #pyautogui.click(game_x+targets[i][0]*square_size+10, game_y+targets[i][1]*square_size+10, button="right")
        # use this opportunity to mark a mine (for subsequent sweeps)
        field[targets[i][0]][targets[i][1]] = -2
        cv2.rectangle(display, (targets[i][0]*disp_cell, targets[i][1]*disp_cell), ((targets[i][0]+1)*disp_cell, (targets[i][1]+1)*disp_cell),color ,-1)
        time.sleep(delay)

def left_clicks (targets, color=(255, 255, 255)):
    delay = 0#.1
    color = (color[2], color[1], color[0])
    global square_size
    global mouse
    global display
    
    #print (targets)
    for i in range (len(targets)):
        mouse.position = (game_x+targets[i][0]*square_size+10, game_y+targets[i][1]*square_size+10)
        mouse.click(Button.left)
        #pyautogui.click(game_x+targets[i][0]*square_size+10, game_y+targets[i][1]*square_size+10)
        # use this opportunity to mark a non-mine (for subsequent sweeps)
        field[targets[i][0]][targets[i][1]] = -3 # this value does not exist, but it will not be counted towards mines and potential mines
        cv2.rectangle(display, (targets[i][0]*disp_cell, targets[i][1]*disp_cell), ((targets[i][0]+1)*disp_cell, (targets[i][1]+1)*disp_cell),color ,-1)
        time.sleep(delay)
        

# Wave 1: Basic strategy
def basic_sweep():
    global field
    global t_sweep
    global t_click
    global has_opening
    click_mines = []
    click_safe = []
    t = time.time()
    has_opening = False # Check if ther is an opening - if not, we'll not run more difficult sweeps
    
    # go through the whole field
    for i in range (wid):
        for j in range (hgt):
            if int(field[i][j])==0:
                has_opening = True
            if int(field[i][j])>0 and int(field[i][j])<9: # this is a number

                n_potential_mines = 0
                n_existing_mines = 0
                all_empty = []

                # for each cell check its neighbours
                for k in range (-1, 2):
                    for l in range (-1, 2):
                        if (k!=0 or l!=0) and i+k>=0 and j+l>=0 and i+k<wid and j+l<hgt: #not self, not out of field

                            # count all mines, closed and both together (potential mines)
                            if field[i+k][j+l] == -1 or field[i+k][j+l] == -2:
                                n_potential_mines += 1
                            if field[i+k][j+l] == -1: # closed
                                all_empty.append((i+k,j+l))
                            if field[i+k][j+l] == -2: # mine
                                n_existing_mines += 1

                # Case 1: Number = potential mines: all closed are mines                    
                if field[i][j] == n_potential_mines:
                    for k in range (len(all_empty)):
                        if all_empty[k] not in click_mines:
                            click_mines.append(all_empty[k])
                            field[all_empty[k][0]][all_empty[k][1]] = -2
                            #print ("Found Mine:", all_empty[k])

                # Case 2: Number = mines: all closed are safe
                if field[i][j] == n_existing_mines:
                    for k in range (len(all_empty)):
                        if all_empty[k] not in click_safe:
                            click_safe.append(all_empty[k])
                            
                all_empty.clear()

    t_sweep += time.time() - t

    t=time.time()
    # Let's do some cliiking            
    right_clicks (click_mines, color_mine_1 )
    #print ("Basic found mines: ", len(click_mines))
    left_clicks (click_safe, color_safe_1 )
    #print ("Basic found safe: ", len(click_safe))
    t_click += time.time() - t
    return len(click_safe)+len(click_mines)
                    
# Wave 2: Advance strategy (a.k.a. Groups)
def advance_sweep():
    global field
    global t_sweep
    global t_click
    
    groups = []
    click_safe = []
    click_mines = []
    t = time.time()
    
    for i in range (wid):
        for j in range (hgt):
            if int(field[i][j])>0 and int(field[i][j])<9: # this is a number

                n_potential_mines = 0
                all_empty = []
                n_existing_mines = 0
                for k in range (-1, 2):
                    for l in range (-1, 2):
                        if (k!=0 or l!=0) and i+k>=0 and j+l>=0 and i+k<wid and j+l<hgt: #not self, not out of field
                            if field[i+k][j+l] == -1 or field[i+k][j+l] == -2:
                                n_potential_mines += 1
                            if field[i+k][j+l] == -1: # closed
                                all_empty.append((i+k,j+l))
                            if field[i+k][j+l] == -2: # mine
                                n_existing_mines += 1
                                
                if field[i][j] != n_potential_mines:
                    groups.append((all_empty, field[i][j]-n_existing_mines))
                    
                #all_empty.clear()
    #print (groups)
    extra_groups = []
    for i in range (len(groups)): # go through cell groups
        for j in range (len(groups)): # and cross reference them with every other element
            if i!=j and groups[i] != groups[j]:
                subgroup_flag = 0 # raises if i is not subgroup of j, not need these ones
                for e in range (len(groups[i][0])):
                    if groups[i][0][e] not in groups[j][0]:
                        subgroup_flag = 1

                # found an inclusive pair, with the same N of mines
                if subgroup_flag == 0 and groups[i][1]==groups[j][1]:
                    #print ("Group" , groups[i][0], "and", groups[j][0], "is it")
                    # diff cells in a larger pair are empty
                    for k in range (len(groups[j][0])):
                        if groups[j][0][k] not in groups[i][0] and groups[j][0][k] not in click_safe:
                            click_safe.append(groups[j][0][k])
                # found an inclusive pair, difference in size = difference in mines
                elif subgroup_flag == 0 and groups[j][1]-groups[i][1]==len(groups[j][0])-len(groups[i][0]):
                    # diff cells in a larger pair are mines
                    #print ("Group" , groups[i][0], "and", groups[j][0], "is it")
                    for k in range (len(groups[j][0])):
                        if groups[j][0][k] not in groups[i][0] and groups[j][0][k] not in click_mines:
                            click_mines.append(groups[j][0][k])
                elif subgroup_flag == 0: #create extra group
                    extra_group = []
                    for k in range (len(groups[j][0])):
                        if groups[j][0][k] not in groups[i][0]:
                            extra_group.append(groups[j][0][k])
                    extra_groups.append((extra_group, groups[j][1]-groups[i][1]))

    #print (extra_groups)

    for i in range (len(extra_groups)): # go through cell groups
        for j in range (len(groups)): # and cross reference them with every other element
            if extra_groups[i] != groups[j]:# and len(extra_groups[i])<len(groups[j]):
                subgroup_flag = 0 # raises if i is not subgroup of j, not need these ones
                for e in range (len(extra_groups[i][0])):
                    if extra_groups[i][0][e] not in groups[j][0]:
                        subgroup_flag = 1

                # found an inclusive pair, with the same N of mines
                if subgroup_flag == 0 and extra_groups[i][1]==groups[j][1]:
                    #print ("Group" , groups[i][0], "and", groups[j][0], "is it")
                    # diff cells in a larger pair are empty
                    for k in range (len(groups[j][0])):
                        if groups[j][0][k] not in extra_groups[i][0] and groups[j][0][k] not in click_safe:
                            click_safe.append(groups[j][0][k])
                            #print ("extra safe A")
                # found an inclusive pair, difference in size = difference in mines
                elif subgroup_flag == 0 and groups[j][1]-extra_groups[i][1]==len(groups[j][0])-len(extra_groups[i][0]):
                    # diff cells in a larger pair are mines
                    #print ("Group" , groups[i][0], "and", groups[j][0], "is it")
                    for k in range (len(groups[j][0])):
                        if groups[j][0][k] not in extra_groups[i][0] and groups[j][0][k] not in click_mines:
                            click_mines.append(groups[j][0][k])
                            #print ("extra mine A")

            if extra_groups[i] != groups[j]:# and len(extra_groups[i])>len(groups[j]):
                subgroup_flag = 0 # raises if i is not subgroup of j, not need these ones
                for e in range (len(groups[j][0])):
                    if groups[j][0][e] not in extra_groups[i][0]:
                        subgroup_flag = 1

                # found an inclusive pair, with the same N of mines
                if subgroup_flag == 0 and extra_groups[i][1]==groups[j][1]:
                    #print ("Group" , groups[i][0], "and", groups[j][0], "is it")
                    # diff cells in a larger pair are empty
                    for k in range (len(extra_groups[i][0])):
                        if extra_groups[i][0][k] not in groups[j][0] and extra_groups[i][0][k] not in click_safe:
                            click_safe.append(extra_groups[i][0][k])
                            #print ("extra safe B")
                # found an inclusive pair, difference in size = difference in mines
                elif subgroup_flag == 0 and extra_groups[i][1]-groups[j][1]==len(extra_groups[i][0])-len(groups[j][0]):
                    # diff cells in a larger pair are mines
                    #print ("Group" , groups[i][0], "and", groups[j][0], "is it")
                    for k in range (len(extra_groups[i][0])):
                        if extra_groups[i][0][k] not in groups[j][0] and extra_groups[i][0][k] not in click_mines:
                            click_mines.append(extra_groups[i][0][k])
                            #print ("extra mine B")


    t_sweep += time.time() - t
    t=time.time()                            

    right_clicks (click_mines, color_mine_2)
    #print ("Advance found mines: ", len(click_mines))
    left_clicks (click_safe, color_safe_2)
    #print ("Advance found safe: ", len(click_safe))
    t_click += time.time() - t
    return len(click_safe) + len(click_mines)

# break a group into "no more than" subgroups
def take_apart_max (group, mines, max_mines):
    if len(group)>=mines+1:
        if (group, mines) not in max_mines:
            max_mines.append((group, mines))
        for i in range (len(group)):
            new_group=[]
            for j in range (len(group)):
                if i!=j:
                    new_group.append(group[j])
            take_apart_max (new_group, mines, max_mines)
    return

# break a group into "at least" subgroups
def take_apart_min (group, mines, min_mines):
    if mines>=1:
        if (group, mines) not in min_mines:
            min_mines.append((group, mines))
        for i in range (len(group)):
            new_group=[]
            for j in range (len(group)):
                if i!=j:
                    new_group.append(group[j])
            take_apart_min (new_group, mines-1, min_mines)
    return
            
# Wave 3: Even more  Advance logic (a.k.a. Subgroups)
def genius_sweep():
    global field
    global t_sweep
    global t_click
    
    min_mines = []
    max_mines = []
    groups = []
    click_safe = []
    click_mines = []
    t = time.time()
    
    for i in range (wid):
        for j in range (hgt):
            if int(field[i][j])>0 and int(field[i][j])<9: # this is a number

                n_potential_mines = 0
                all_empty = []
                n_existing_mines = 0
                for k in range (-1, 2):
                    for l in range (-1, 2):
                        if (k!=0 or l!=0) and i+k>=0 and j+l>=0 and i+k<wid and j+l<hgt: #not self, not out of field
                            if field[i+k][j+l] == -1 or field[i+k][j+l] == -2:
                                n_potential_mines += 1
                            if field[i+k][j+l] == -1: # closed
                                all_empty.append((i+k,j+l))
                            if field[i+k][j+l] == -2: # mine
                                n_existing_mines += 1

                if field[i][j] != n_potential_mines:
                    groups.append((all_empty, field[i][j]-n_existing_mines))
                # subgroup A: has at least N mines
                if len(all_empty)>field[i][j]+1 and len(all_empty) != 8: # dont use it for cells "out in the field"
                    #print ("For sub A: ", all_empty, field[i][j] )
                    take_apart_max (all_empty, field[i][j], max_mines)
                    #print (max_mines)
                # subgroup B: has no more then N mines
                if len(all_empty)>field[i][j] and field[i][j]>1  and len(all_empty) != 8:
                    #print ("For sub B: ", all_empty, field[i][j] )
                    take_apart_min (all_empty, field[i][j]-n_existing_mines, min_mines)
                    #print (min_mines)
                    
    for i in range (len(max_mines)): # go through all the broken down groups A
        for j in range (len(groups)): # and cross reference them with every other element
            if max_mines[i][0] != groups[j][0]:
                subgroup_flag = 0
                difference = 0
                for e in range (len(max_mines[i][0])):
                    if max_mines[i][0][e] not in groups[j][0]:
                        subgroup_flag = 1
                    else:
                        difference += 1
                        
                # found an inclusive pair, with the same N of mines
                if subgroup_flag == 0 and max_mines[i][1]+len(groups[j][0])-len(max_mines[i][0])==groups[j][1]:
                    #print ("Found: ", max_mines[i], "and", groups[j])
                    #diff cells in a larger pair are mines
                    for k in range (len(groups[j][0])):
                        if groups[j][0][k] not in max_mines[i][0] and groups[j][0][k] not in click_mines:
                            click_mines.append(groups[j][0][k])

    for i in range (len(min_mines)): # go through all the broken down groups B
        for j in range (len(groups)): # and cross reference them with every other element
            if min_mines[i][0] != groups[j][0]:
                subgroup_flag = 0
                difference = 0
                for e in range (len(min_mines[i][0])):
                    if min_mines[i][0][e] not in groups[j][0]:
                        subgroup_flag = 1
                    else:
                        difference += 1
                        
                # found an inclusive pair, with 0 mines left                       
                if subgroup_flag == 0 and groups[j][1]==min_mines[i][1]:
                    #print ("Found: ", min_mines[i], "and", groups[j])
                    #diff cells in a larger pair are safe
                    for k in range (len(groups[j][0])):
                        if groups[j][0][k] not in min_mines[i][0] and groups[j][0][k] not in click_safe:
                            click_safe.append(groups[j][0][k])

    t_sweep += time.time() - t
    t=time.time()
    
    right_clicks (click_mines, color_mine_3)
    #print ("Genius found mines: ", len(click_mines))
    left_clicks (click_safe, color_safe_3)
    #print ("Genius found safe: ", len(click_safe))
    t_click += time.time() - t
    return len(click_safe) + len(click_mines)

# Count strategy (basic). Just if remaining cells = remaining mines
def counter_sweep():
    global t_sweep
    global t_click
    click_mines = []
    click_safe = []
    t = time.time()

    total_closed = 0
    all_closed = []
    # go through the hole field
    for i in range (wid):
        for j in range (hgt):
            if int(field[i][j])==-1: # this is a closed cell
                total_closed += 1
                all_closed.append((i,j))
    total_mines = get_mine_counter()
    if total_mines == 0:
        click_safe = all_closed
    if total_mines == total_closed:
        click_mines = all_closed

    t_sweep += time.time() - t
    t=time.time()
    
    right_clicks (click_mines, color_counter)
    left_clicks (click_safe, color_counter)
    t_click += time.time() - t
    return len(click_safe) + len(click_mines)

# more advance counter logic:
# if mines tied in those groups, remaining cells are safe
def counter_sweep_adv():
    global t_sweep
    global t_click
    click_mines = []
    click_safe = []
    t = time.time()
    
    total_mines = get_mine_counter()
    #print (total_mines)
    groups = []
    total_closed = 0
    all_closed = []
    for i in range (wid):
        for j in range (hgt):
            if int(field[i][j])==-1: # this is a closed cell
                total_closed += 1
                all_closed.append((i,j))
            if int(field[i][j])>0 and int(field[i][j])<9: # this is a number
                n_potential_mines = 0
                all_empty = []
                n_existing_mines = 0
                for k in range (-1, 2):
                    for l in range (-1, 2):
                        if (k!=0 or l!=0) and i+k>=0 and j+l>=0 and i+k<wid and j+l<hgt: #not self, not out of field
                            if field[i+k][j+l] == -1 or field[i+k][j+l] == -2:
                                n_potential_mines += 1
                            if field[i+k][j+l] == -1: # closed
                                all_empty.append((i+k,j+l))
                            if field[i+k][j+l] == -2: # mine
                                n_existing_mines += 1
                if field[i][j] != n_potential_mines:
                    groups.append((all_empty, field[i][j]-n_existing_mines))
    #print (groups)
    coverage = []
    coverage_n = 0
    for i in range(len(groups)*10): # no reason, just seems to be an appropriate N of times to shuffle
        if i!=0:
            random.shuffle(groups) 
        coverage = []
        coverage_n = 0
        for group in groups:
            overlap=False
            for cell in group[0]:
                if cell in coverage:
                    overlap=True
            if not overlap:
                coverage += group[0]
                coverage_n += group[1]
        if total_mines == coverage_n and len(all_closed)>len(coverage) \
            or total_mines - coverage_n == len(all_closed)-len(coverage):
            break
            
    #print (coverage)
    #print (coverage_n)

    # all mines are tied, safe cells remain
    if total_mines == coverage_n and len(all_closed)>len(coverage):
        for i in range(len(all_closed)):
            if all_closed[i] not in coverage:
                click_safe.append (all_closed[i])
                #print ("cover safe")

    # all not ties == remaining closed => are mines
    if total_mines - coverage_n == len(all_closed)-len(coverage):
        for i in range(len(all_closed)):
            if all_closed[i] not in coverage:
                click_mines.append (all_closed[i])
                #print ("cover mines")

    
    t_sweep += time.time() - t
    t=time.time()
    
    right_clicks (click_mines, color_counter)
    left_clicks (click_safe, color_counter)
    t_click += time.time() - t
    return len(click_safe) + len(click_mines)
               
# Constraint Satisfaction Problem approach
def csp_sweep():
    global t_sweep
    global t_click
    click_mines = []
    click_safe = []
    t = time.time()
    total_mines = get_mine_counter()
    groups = []
    
    for i in range (wid):
        for j in range (hgt):
            if int(field[i][j])>0 and int(field[i][j])<9: # this is a number

                n_potential_mines = 0
                all_empty = []
                n_existing_mines = 0
                for k in range (-1, 2):
                    for l in range (-1, 2):
                        if (k!=0 or l!=0) and i+k>=0 and j+l>=0 and i+k<wid and j+l<hgt: #not self, not out of field
                            if field[i+k][j+l] == -1 or field[i+k][j+l] == -2:
                                n_potential_mines += 1
                            if field[i+k][j+l] == -1: # closed
                                all_empty.append((i+k,j+l))
                            if field[i+k][j+l] == -2: # mine
                                n_existing_mines += 1
                                
                if field[i][j] != n_potential_mines:
                    groups.append((all_empty, field[i][j]-n_existing_mines))
                    

    gr = groups.copy()
    # create clusters
    clusters = []
    while len (gr) > 0:
        clusters.append([])
        clusters[-1] += gr.pop(0)[0]
        still_going = True
        while still_going:
            still_going = False
            for i in range(len(gr))[::-1]:
                for j in range(len(gr[i][0])):
                    if gr[i][0][j] in clusters[-1]:
                        clusters[-1] += gr.pop(i)[0]
                        still_going = True
                        break
    # dedup them
    #print ("CSP: ", end="")
    for i in range(len(clusters)):
        clusters[i] = list(set(clusters[i]))
        #print (len(clusters[i]), end=", ")
    #print ("\n", clusters)


    
    for cluster in clusters: # let analyze!
        if len(cluster)<21: # to prevent computational explosions
            #print ("="*40, "\nCluster:", cluster)
            
            # create conditions: (mask, mines)
            # mask is binary where 1 are cells in cluster to check
            conditions = []
            for group in groups:
                condition_mask = 0
                for cell in group[0]:
                    if cell in cluster:
                        condition_mask += 2**cluster.index(cell)
                if condition_mask != 0 and (condition_mask, group[1]) not in conditions:
                    conditions.append ((condition_mask, group[1]))
                    #print (group, "{0:b}".format(condition_mask), group[1])
            #print ("Conditions:", conditions)

            # generate all permutations, check against conditions
            # find solutions that fit
            solutions = []
            for i in range(2**len(cluster)):
                for condition in conditions:
                    # check i for condition 
                    if bin(condition[0] & i).count('1') != condition[1]:
                        break
                else: # passed all conditions
                    if bin(i).count('1') <= total_mines:
                        solutions.append(i)
            #print ("Solutions:", solutions)

            # look for "always mines" and "always safe" in solutions
            safe_mask = 0
            mine_mask = 2**len(cluster)-1
            # by running two bitmasks through all solutions
            for solution in solutions:
                #print (format(solution, '#016b'))
                safe_mask |= solution
                mine_mask &=solution
            # and then checking if there are 0s in safe mask and 1s in mine mask
            for i in range(2, len(cluster)+2): # 2, because string representation starts with "0b"
                #print (format(safe_mask, '#0'+str(len(cluster)+2)+'b')[i])
                if format(safe_mask, '#0'+str(len(cluster)+2)+'b')[i]=="0":
                    click_safe.append(cluster[-i+1])
                if format(mine_mask, '#0'+str(len(cluster)+2)+'b')[i]=="1":
                    click_mines.append(cluster[-i+1])
                    
            #print (format(safe_mask, '#0'+str(len(cluster)+2)+'b'), "Safe mask")
            #print (format(mine_mask, '#0'+str(len(cluster)+2)+'b'), "Mine mask")
        else:
            pass
            #print ("!!! Skipping Clister the size of:", len(cluster))
    #print (click_safe)        
    #print (click_mines)        
    #print ("CSP Time:", time.time() - t)        
                           
    t_sweep += time.time() - t
    t=time.time()
    
    right_clicks (click_mines, color_csp)
    left_clicks (click_safe, color_csp)
    t_click += time.time() - t
    
    return len(click_safe) + len(click_mines)                

                    
        
        



    
# last resort - click the random safe (but the one most likely to be safe)
def random_sweep():
    groups = []
    global field
    global t_sweep
    global t_click
    global last_guessing
    click_safe = []
    all_closed = []
    total_closed = 0
    t = time.time()
    new_board = 1 # flag to tell if the game just started
    # go through the hole field
    for i in range (wid):
        for j in range (hgt):
            if int(field[i][j]) == -1:
                total_closed += 1 # count all empties
                all_closed.append((i,j)) # look at all closed cells and count their probability
            if int(field[i][j])>0 and int(field[i][j])<9: # this is a number

                n_potential_mines = 0
                n_existing_mines = 0
                all_empty = []
                new_board = 0

                # for each cell check its neighbours
                for k in range (-1, 2):
                    for l in range (-1, 2):
                        if (k!=0 or l!=0) and i+k>=0 and j+l>=0 and i+k<wid and j+l<hgt: #not self, not out of field

                            # count all mines, closed and both together (potential mines)
                            if field[i+k][j+l] == -1 or field[i+k][j+l] == -2:
                                n_potential_mines += 1
                            if field[i+k][j+l] == -1: # closed
                                all_empty.append((i+k,j+l))
                            if field[i+k][j+l] == -2: # mine
                                n_existing_mines += 1

                # create a list of all fieldsets with their mine probability                
                if field[i][j] != n_potential_mines:
                    groups.append((all_empty, field[i][j]-n_existing_mines, (field[i][j]-n_existing_mines) / len(all_empty)))


    # add all cells and their probability
    if  len(all_closed) > 0:
        total_mines = get_mine_counter()
        #groups.append((all_closed, total_mines, total_mines / len(all_closed)))

    coverage = []
    coverage_n = 0
    best_coverage = []
    best_coverage_n = 0
    for i in range(len(groups)*10): # no reason, just seems to be an appropriate N of times to shuffle
        if i!=0:
            random.shuffle(groups) 
        coverage = []
        coverage_n = 0
        for group in groups:
            overlap=False
            for cell in group[0]:
                if cell in coverage:
                    overlap=True
            if not overlap:
                coverage += group[0]
                coverage_n += group[1]
        if coverage_n > best_coverage_n:
            best_coverage_n = coverage_n
            best_coverage = coverage.copy()
    #print ("Coverage:", best_coverage_n, "mines in", len(best_coverage), "cells")

    other_cells = [] # those that cant be counted and tied
    for cell in all_closed:
        if cell not in best_coverage:
            other_cells.append(cell)
    if len(other_cells) > 0:
        groups.append((other_cells, total_mines-best_coverage_n, (total_mines-best_coverage_n) / len(other_cells)))        
    #print ("Other chance:", (total_mines-best_coverage_n)  / len(other_cells))        
        
    #print (groups)
    # go through that list, finding the lowest ones

    all_chances = np.zeros((hgt,wid))
    
    lowest_chance = 1
    safest_cells = []
    for i in range (wid):
        for j in range (hgt):
            if (i,j) in all_closed:
                count_chances = [0,]
                for k in range (len(groups)):
                    if (i,j) in groups[k][0]:
                        count_chances.append(groups[k][2])
                all_chances[j,i] = max(count_chances)
                if max(count_chances) < lowest_chance:
                    lowest_chance = max(count_chances)
                    safest_cells = []
                    safest_cells.append((i,j))
                if max(count_chances) == lowest_chance and (i,j) not in safest_cells:
                    safest_cells.append((i,j))
#    print (all_chances)
    #print ("Lowest mine chance:", lowest_chance)
#    print ("Safest cells:", safest_cells)

    max_open_chance = 0
    max_open_cells = []
    for cell in safest_cells:
        open_chance = 1 - all_chances[cell[1],cell[0]]
        for k in range (-1, 2):
            for l in range (-1, 2):
                if (k!=0 or l!=0) and cell[0]+k>=0 and cell[1]+l>=0 and cell[0]+k<wid and cell[1]+l<hgt: #not self, not out of field
                    open_chance = open_chance * (1 - all_chances[cell[1]+l,cell[0]+k])
        if open_chance == max_open_chance:
            max_open_cells.append(cell)
        if open_chance > max_open_chance:
            max_open_chance = open_chance
            max_open_cells = []
            max_open_cells.append(cell)

        #print (open_chance)
    #print ("Max open chance: ", max_open_chance)        
#    print ("Cells with max open chance", max_open_cells)        

    if len(max_open_cells)>0:
        safest_cell = max_open_cells[random.randint(0,len(max_open_cells)-1)]

        
        #print ("Random cell: ", safest_cell)
    elif lowest_chance == 1 and new_board == 0:
        if total_closed > 0:
            #print ("Stuck")
            return -10
        else:
            #print ("All open")
            return 0

    last_guessing = lowest_chance
    color = (color_random[0]*2*lowest_chance, 0, 0)
    #time.sleep(0.3)
    #im.save(str(time.time())+".png")
    #print ("Safest cell: ", safest_cell, lowest_chance)
        
    t_sweep += time.time() - t
    t=time.time()
    
    left_clicks ((safest_cell,), color)
    t_click += time.time() - t
    return 1

# get a cell in an indicator
def get_digit_pic(d):
    global im
    p = 2-d # place from the left
    box = (game_x + 7 + p*16, game_y-48, game_x + 7 + (p+1)*16-1, game_y-21)
    region = im.crop(box)
    #region.save("counter"+str(d)+".jpg")
    return region

# read if segment is on
def segment_is_on(im, n):
    seg = [(5,0,10,3), (1,4,3,10), (11,4,13,10), (5,12,10,15), (1,17,3,22), (11,17,13,22), (5,24,10,26)]
    px = im.load()
    sum = 0
    for i in range(seg[n][0], seg[n][2]+1):
        for j in range(seg[n][1], seg[n][3]+1):
            sum += px[i,j][0]
    if sum>3000:
        return 1
    return 0
        
def read_digit_pic(im):
    lit_segments = []
    for i in range(7):
        lit_segments.append( segment_is_on(im, i) )
    if lit_segments == [1, 1, 1, 0, 1, 1, 1]:
        return 0
    if lit_segments == [0, 0, 1, 0, 0, 1, 0]:
        return 1
    if lit_segments == [1, 0, 1, 1, 1, 0, 1]:
        return 2
    if lit_segments == [1, 0, 1, 1, 0, 1, 1]:
        return 3
    if lit_segments == [0, 1, 1, 1, 0, 1, 0]:
        return 4
    if lit_segments == [1, 1, 0, 1, 0, 1, 1]:
        return 5
    if lit_segments == [1, 1, 0, 1, 1, 1, 1]:
        return 6
    if lit_segments == [1, 0, 1, 0, 0, 1, 0]:
        return 7
    if lit_segments == [1, 1, 1, 1, 1, 1, 1]:
        return 8
    if lit_segments == [1, 1, 1, 1, 0, 1, 1]:
        return 9
  
# Get the number of remaining mines
def get_mine_counter():
    mines = 0
    for d in range(3): # 0 is singles, 2 hundreds
        mines += read_digit_pic( get_digit_pic(d) ) * 10 ** d 
    return mines

# One cycle - from a screenshot to clicking stuff                
def cycle():                
    global im
    global px
    global t_screen
    global t_read
    global field
    global s_basic
    global s_advance
    global s_genious
    global s_counter
    global s_csp
    global s_random

    t = time.time()
    
    im = pyautogui.screenshot()
    #im = Image.open("screen3.png")
    px = im.load()
    t_screen += time.time() - t

    t = time.time()
    result = read_data()
    if result == -999:
        return -999
    if result == -100:
        return -100
    t_read += time.time() - t
    #nice_print()    
    basic = basic_sweep()
    advance = advance_sweep()
    if has_opening:
        genious = genius_sweep()
    else:
        genious = 0 # otherwise too much load in the beginning
    found = basic + advance + genious
    s_basic += basic
    s_advance += advance
    s_genious += genious

    
    
    if found == 0:
        #print (get_mine_counter())
        counter = counter_sweep()
        counter += counter_sweep_adv()
        found += counter
        s_counter += counter
        if found == 0:
            csp = csp_sweep()
            found += csp
            s_csp += csp
            if found == 0:
                random = random_sweep()
                found += random
                s_random += random

    field.clear()
    return found

# window for a fancy graphics
display = np.zeros((100,100,3), np.uint8)
disp_cell=20

# Playing 1 game
def main():
    global im
    global px
    global games_total
    global games_won
    global display
    
    
    
    im = pyautogui.screenshot()
    px = im.load()
    grey_square = find_grey_square()
    if not grey_square:
        print ("Can't find the game")
        return
    #print ("Grey square: ", grey_square)
    find_grid(grey_square)
    #print(game_x, game_y, wid, hgt)

    
    display = np.zeros((hgt*disp_cell,wid*disp_cell,3), np.uint8)
    cv2.imshow('Strategy type map',display)
    cv2.moveWindow('Strategy type map', 1194,600);
    cv2.waitKey(20)
        
    games_total += 1
    print ("="*40)
    print ("Starting game: ", games_total)
    result = cycle()
    while result>0:
        result = cycle()
        #print ("")
        cv2.imshow('Strategy type map',display)
        cv2.waitKey(20)
        
    # result and stats
    if result == -999:
        print ("Can't find the field")
        return -999
    if result == -100:
        cv2.rectangle(display, (0,0), (wid*disp_cell, hgt*disp_cell),(0,0,255) ,10)
        print ("Lost")
        print ("Last guess: ", last_guessing)
    if result == -10:
        print ("Stuck")
    if result == 0:
        print ("Won")
        cv2.rectangle(display, (0,0), (wid*disp_cell, hgt*disp_cell),(0,255,0) ,10)
        games_won += 1
    s_total = s_basic + s_advance + s_genious + s_counter + s_csp + s_random
##    print ("")
    print ("Wins: ", games_won, games_won / games_total)
    success_rate.append(games_won / games_total)
##    print ("")
##    print ("Basic sweep: ", s_basic, s_basic / s_total)
##    print ("Group sweep: ", s_advance, s_advance / s_total)
##    print ("Subgroup sweep: ", s_genious, s_genious / s_total)
##    print ("Counter sweep: ", s_counter, s_counter / s_total)
##    print ("CSP sweep: ", s_csp, s_csp / s_total)
##    print ("Random sweep: ", s_random, s_random / s_total)
##    print ("")
##    print ("Screenshot time: ", t_screen)
##    print ("Image processing time: ", t_read)
##    print ("Searching mines time: ", t_sweep)
##    print ("Clicking time: ", t_click)
    cv2.imshow('Strategy type map',display)
    cv2.waitKey(20)
    time.sleep(0.6)

# Playing inf, 100, 1000 games 
def infinite():
    play_n(1000000000)

def play_n(n):
    keyboard.send('f2')
    time.sleep(0.5)
    for i in range(n):
        keyboard.send('f2')
        time.sleep(0.2)
        if main() == -999:
            break
        dispaly_diagrams()
        time.sleep(0.2)
    time.sleep(15)
    cv2.destroyAllWindows()

def play_100():
    play_n(100)
    

def play_1000():
    play_n(1000)

def reset_data():
    global games_total
    global games_won
    global s_basic
    global s_advance
    global s_genious
    global s_counter
    global s_random
    global t_screen
    global t_read
    global t_sweep
    global t_click
    global games_total
    global games_won
    global s_basic
    global s_advance
    global s_genious
    global s_counter
    global s_csp
    global s_random
    global t_screen
    global t_read
    global t_sweep
    global t_click
    games_total  = 0
    games_won    = 0
    s_basic      = 0
    s_advance    = 0
    s_genious    = 0
    s_counter    = 0
    s_csp        = 0
    s_random     = 0
    t_screen     = 0
    t_read       = 0
    t_sweep      = 0
    t_click      = 0
    print ("Data has been reset")

# Success rate graph
def dispaly_diagrams():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1,len(success_rate)+1), success_rate)
    ax.set_ylim([0,1.1])
    ax.set(xlabel='attempts', ylabel='Success rate', title="{:.1%}".format(games_won / games_total))
    ax.grid()
    fig.savefig("success.png")
    plt.close(fig)
    diagimg = cv2.imread('success.png')
    cv2.imshow("success rate", diagimg)
    cv2.waitKey(10)
    #plt.show()
 
im = Image.new("RGB", (100, 100), "white")
px = im.load()

## Test for the cell image
##im = pyautogui.screenshot()
##px = im.load()
##grey_square = find_grey_square()
##print (grey_square)
##find_grid(grey_square)
##cell = get_cell(0,1)
##cell.show()
##code = ave_cell(cell)
##print (code)

# for new strategy
##im = Image.open("rand3.png")
##px = im.load()
##grey_square = find_grey_square()
##find_grid(grey_square)
##read_data()            
###nice_print()
##random_sweep()
    
print ("F3 - play now (once)")
print ("F4 - play forever")
print ("F5 - play 100 times")
print ("F6 - play 1000 times")
print ("F10 - reset data")

    
keyboard.add_hotkey('f3', main)
keyboard.add_hotkey('f4', infinite)
keyboard.add_hotkey('f5', play_100)
keyboard.add_hotkey('f6', play_1000)
keyboard.add_hotkey('f10', reset_data)
keyboard.wait('esc')

