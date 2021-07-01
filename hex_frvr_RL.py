import gym
import queue, threading, time
from pynput.keyboard import Key, Listener
from PIL import ImageGrab
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.transforms import Affine2D
from matplotlib import pyplot, transforms
import matplotlib
import mpl_toolkits.axisartist.floating_axes as floating_axes
import random
from IPython import display


def create_board():
    board = []
    occupied = {}
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if(i+j+k==0):
                    board.append([j,k,i])
                    occupied[''.join(str(idx) for idx in [j,k,i])] = 0
    return board, occupied

def check_valid(occupied, shape, location):
        
    if(len(shape) == 1 and type(shape[0][0]) is not int):
        shape = shape[0]
    loc = ''.join(str(idx) for idx in location) 
    if(shape[0] == [0,0,0] and shape[1] == [0,0,0] and occupied[loc] == 0):
        return True



    if(occupied[loc] == 1):
        return False
    for hex in shape:
        place = ''.join(str(location[idx] + loc_hex) for idx, loc_hex in enumerate(hex))
        if(place not in occupied):
            return False
        if(occupied[place]) == 1:
            return False
    return True


def remove_lines_filled(occupied):
    score = 40
    global current_score
    rows_to_remove = []
    multiplier = 1
    row_lens = []
    axises = [0, 1, 2]
    size = 4
    for axis in axises:
        for row in range(-size, size+1):
            current_row = [x for x in board if x[axis] == row ]
            tile_strs = []
            for tile in current_row:
                tile_strs.append(''.join(str(tl) for tl in tile))
            if(all([occupied[tile_str] == 1 for tile_str in tile_strs])):
                rows_to_remove.append(tile_strs)
                #for tile_str in tile_strs:
                #    occupied[tile_str] = 0
                row_lens.append(len(current_row))
    base = 10 * (len(row_lens) + 1)
    for i, row_len in enumerate(row_lens):
        # Number of pieces * (base score * 1.2^row index)
        score += int(row_len * (math.floor((base * max(1.2**i, 1)))))
    for row in rows_to_remove:
        for tile_str in row:
            occupied[tile_str] = 0
    current_score += score
    return occupied, row_lens

def place_shape_search(occupied, shape, location):
    if(check_valid(occupied, shape, location)):
        if(len(shape) == 1 and type(shape[0][0]) is not int):
            shape = shape[0]
        for hex in shape:
            place = ''.join(str(loc_hex + location[idx]) for idx, loc_hex in enumerate(hex))
            occupied[place] = 1
    return occupied

def place_shape(occupied, shape, location):
    if(check_valid(occupied, shape, location)):
        if(len(shape) == 1 and type(shape[0][0]) is not int):
            shape = shape[0]
        for hex in shape:
            place = ''.join(str(loc_hex + location[idx]) for idx, loc_hex in enumerate(hex))
            occupied[place] = 1
        occupied, row_lens = remove_lines_filled(occupied)
    return occupied, row_lens
        
def possible_moves(occupied, shape):
    valid_moves = []
    for tile in board:
        if(check_valid(occupied, shape, tile)):
            valid_moves.append(tile)
    return valid_moves

def draw_board_with_shapes(board, occupied, current_shapes):

    coord = board
    colors = ["blue" if x == 0 else "green" for x in occupied.values()]
    
    print(current_shapes)
    for i in range(len(current_shapes)):
        if(len(current_shapes[i])) == 1 and len(current_shapes[i][0]) > 1:
            current_shapes[i] = current_shapes[i][0]
    for i in range(len(current_shapes)):
        if(len(current_shapes[i]) == 1 and len(current_shapes[i][0]) == 1):
            current_shapes[i] = [[0,0,0]]


    

# Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

# Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]



    
    shape = current_shapes[0]
    hcoord_shape1 = [c[0] + 9 for c in shape]
    vcoord_shape1 = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. + 6 for c in shape]
    
    shape2 = current_shapes[1]
    hcoord_shape2 = [c[0] + 9 for c in shape2]
    vcoord_shape2 = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in shape2]

    shape3 = current_shapes[2]
    hcoord_shape3 = [c[0] + 9 for c in shape3]
    vcoord_shape3 = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. - 6 for c in shape3]

    for i in range(len(vcoord)):
        temp = vcoord[i]
        vcoord[i] = -hcoord[i]
        hcoord[i] = temp
        
    for i in range(len(vcoord_shape1)):
        temp = vcoord_shape1[i]
        vcoord_shape1[i] = -hcoord_shape1[i]
        hcoord_shape1[i] = temp
        
    for i in range(len(vcoord_shape2)):
        temp = vcoord_shape2[i]
        vcoord_shape2[i] = -hcoord_shape2[i]
        hcoord_shape2[i] = temp
        
    for i in range(len(vcoord_shape3)):
        temp = vcoord_shape3[i]
        vcoord_shape3[i] = -hcoord_shape3[i]
        hcoord_shape3[i] = temp
        


    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.set_aspect('equal')

    for x, y, c in zip(hcoord, vcoord, colors):
        color = c[0]
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3, 
                             orientation=np.radians(120), facecolor = color,
                             alpha=0.3, edgecolor='k')
        ax.add_patch(hex)
        
    for x, y, c in zip(hcoord_shape1, vcoord_shape1, colors):
        color = c[0]
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3, 
                             orientation=np.radians(60), facecolor = "blue",
                             alpha=0.3, edgecolor='k')
        ax.add_patch(hex)
    
    for x, y, c in zip(hcoord_shape2, vcoord_shape2, colors):
        color = c[0]
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3, 
                             orientation=np.radians(60), facecolor = "blue",
                             alpha=0.3, edgecolor='k')
        ax.add_patch(hex)
    
    for x, y, c in zip(hcoord_shape3, vcoord_shape3, colors):
        color = c[0]
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3, 
                             orientation=np.radians(60), facecolor = "blue",
                             alpha=0.3, edgecolor='k')
        ax.add_patch(hex)
        # Also add a text label
    ax.set_xlim([-10,10])
    ax.set_ylim([-16,10])
    display.clear_output(wait=True)
    ax.scatter(hcoord, vcoord, alpha=0.3)
    plt.show()

def draw_available_shapes(board, current_shapes):
    coord = current_shapes[1]
    color = "blue"
    hcoord = [c[0] for c in coord]
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_aspect('equal')

    # Add some coloured hexagons
    for x, y in zip(hcoord, vcoord):
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3, 
                             orientation=np.radians(120), facecolor = color,
                             alpha=0.3, edgecolor='k')
        ax.add_patch(hex)
        # Also add a text label

    # Also add scatter points in hexagon centres
    display.clear_output(wait=True)
    #display.display(ax.scatter(hcoord, vcoord, alpha=0.3))
    ax.scatter(hcoord, vcoord, alpha=0.3)
    plt.show()

def check_moves_left(occupied, shapes):
    for shape in shapes:
        if(len(possible_moves(occupied, shape)) > 0):
            return True
    return False

def generate_board_pos():
    board_pos = np.zeros((9,9,9))
    for tile in board:
        loc = ''.join([str(idx) for idx in tile])
        if(occupied[loc] == 1):
            board_pos[tile[0]][tile[1]][tile[2]] = 1
    return board_pos

def no_of_lines_filled_and_score(occupied):
    multiplier = 0
    score = 40
    row_lens = []
    axises = [0, 1, 2]
    size = 4
    for axis in axises:
        for row in range(-size, size+1):
            current_row = [x for x in board if x[axis] == row ]
            tile_strs = []
            for tile in current_row:
                tile_strs.append(''.join(str(tl) for tl in tile))
            if(all([occupied[tile_str] == 1 for tile_str in tile_strs])):
                row_lens.append(len(current_row))
    base = 10 * (len(row_lens) + 1)
    for i, row_len in enumerate(row_lens):
        # Number of pieces * (base score * 1.2^row index)
        score += int(row_len * (math.floor((base * max(1.2**i, 1)))))
    return len(row_lens), score

def no_outside(occupied):
    no_holes = 0
    for tile in board:
        no_filled_neighbors = 0
        loc = ''.join([str(idx) for idx in tile])
        if(occupied[loc]) == 1:
            no_filled_neighbors = 0
            x_loc = tile[0]
            y_loc = tile[1]
            z_loc = tile[2]
            num_of_neigh_filled = 0
            neigh_locs = [[1, -1, 0], [1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1]]
            for x_new, y_new, z_new in neigh_locs:
                neigh_coords = [x_loc + x_new, y_loc + y_new, z_loc + z_new]
                neigh_loc = ''.join([str(idx) for idx in neigh_coords])
                if(neigh_loc in occupied):    
                    if(occupied[neigh_loc] == 0):
                        no_filled_neighbors += 1
            if(no_filled_neighbors >= 0):
                no_holes += 1
    return no_holes 

def no_holes(occupied):
    no_holes = 0
    for tile in board:
        no_filled_neighbors = 0
        loc = ''.join([str(idx) for idx in tile])
        if(occupied[loc]) == 0:
            no_filled_neighbors = 0
            x_loc = tile[0]
            y_loc = tile[1]
            z_loc = tile[2]
            num_of_neigh_filled = 0
            neigh_locs = [[1, -1, 0], [1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1]]
            for x_new, y_new, z_new in neigh_locs:
                neigh_coords = [x_loc + x_new, y_loc + y_new, z_loc + z_new]
                neigh_loc = ''.join([str(idx) for idx in neigh_coords])
                if(neigh_loc in occupied):    
                    if(occupied[neigh_loc] == 1):
                        no_filled_neighbors += 1
            if(no_filled_neighbors >= 5):
                no_holes += 1
    return no_holes 

def no_tiles_filled(occupied):
    no_filled = 0
    for value in occupied.values():
        no_filled += value
    return no_filled

possible_shapes=np.asarray([[[-1,0,1],[0,0,0],[1,0,-1],[2,0,-2]],[[-1,1,0],[0,0,0],[1,-1,0],[2,-2,0]],[[0,-1,1],[0,0,0],[0,1,-1],[0,2,-2]],[[0,-1,1],[0,0,0],[0,1,-1],[1,-1,0]],[[0,-1,1],[0,0,0],[0,1,-1],[-1,1,0]],[[-1,1,0],[0,0,0],[1,-1,0],[0,1,-1]],[[-1,1,0],[0,0,0],[1,-1,0],[0,-1,1]],[[-1,0,1],[0,0,0],[1,0,-1],[-1,1,0]],[[-1,0,1],[0,0,0],[1,0,-1],[1,-1,0]],[[0,0,0],[-1,1,0],[-1,0,1],[0,-1,1]],[[0,0,0],[-1,1,0],[1,0,-1],[0,1,-1]],
                           [[0,0,0],
                            [-1, 1, 0],
                            [1, 0, -1],
                            [1, 1, -2]],
                            [[0,0,0],
                            [1, 0, -1],
                            [0, 1, -1],
                            [0, -1, 1]],
                            [[0,0,0],
                            [1,-1,0],
                            [-1,0,1],
                            [-1, -1, 2]],
                            [[0,0,0],
                            [0,1,-1],
                            [1,0,-1],
                            [2,-1,-1]],
                            [[0,0,0],
                            [1,-1,0],
                            [2,-2,0],
                            [2,-1,-1]],
                            [[-1,1,0],
                            [0,1,-1],
                            [1,0,-1],
                            [1,-1,0]],
                            [[-1,1,0],
                            [-1,0,1],
                            [0,-1,1],
                            [1,-1,0]],
                            [[0,0,0], [0,0,0]],
                            [[0,0,0], [0,0,0]],
                            [[0,0,0], [0,0,0]],
                            [[0,0,0], [0,0,0]],
                           ])

                           
def get_params(occupied, available_shapes, shape_idx):
    pos_moves_cnt = 0
    for o_shape in possible_shapes:
        pos_moves_cnt += len(possible_moves(occupied, o_shape))
    rows_filled, score = no_of_lines_filled_and_score(occupied)
    no_outside_v = no_outside(occupied)
    no_holes_v = no_holes(occupied)
    tiles_filled = no_tiles_filled(occupied)
    possible_other_shape = 0
    other_shapes = [x for idx, x in enumerate(available_shapes) if idx != shape_idx]

    top_next_move = 0
    for shape in other_shapes:
        pos_moves = possible_moves(occupied, shape)
        if(len(pos_moves)) > 0:
            possible_other_shape += 1
            if(len(pos_moves) < 15):
                for move in pos_moves:
                    occpd = place_shape_search(copy.deepcopy(occupied), shape, move)
                    prms = get_params_in_advance(occpd)
                    if(prms > top_next_move):
                        top_next_move = prms
            
    if(top_next_move < -3000):
        print(top_next_move)
    return [pos_moves_cnt/50, rows_filled/27, score/150, no_outside_v/61, no_holes_v/61, tiles_filled/61, possible_other_shape, top_next_move]
    
    
def get_params_in_advance(occupied):
    rows_filled, score = no_of_lines_filled_and_score(occupied)
    no_outside_v = no_outside(occupied)
    no_holes_v = no_holes(occupied)
    tiles_filled = no_tiles_filled(occupied)
    
    return weights[3]*score/150 + weights[4]*no_outside_v/61 + weights[5] * tiles_filled/61

def weight(occupied, shape, shape_idx, location, shapes):
    #draw_board(board, occupied)
    occupied = place_shape_search(copy.deepcopy(occupied), shape, location)
    params = get_params(occupied, shapes, shape_idx)
    return weights[0] * params[0] + weights[1] * params[1] + weights[2] * params[2] + weights[3] * params[3] + weights[4] * params[4] + weights[5]*params[5]+ weights[6]*params[6]+weights[7]*params[7],params
    
             
import copy

def get_best_move(occupied, shapes):
    if(epsilon > random.random()):
        pos_random = []
        for idx, shape in enumerate(shapes):
            pos_moves = possible_moves(occupied, shape)
            pos_moves = [((shape, x), idx) for x in pos_moves]
            pos_random.extend(pos_moves)
        choice = random.choice(pos_random)
        rand_occupied = place_shape_search(copy.deepcopy(occupied), choice[0][0], choice[0][1])
        params = get_params(rand_occupied, shapes, choice[0][1])
        return choice[0], choice[1], params
    else:
        top_weight = -999999999
        top_move = None
        top_shape_idx = -1
        top_params = None
        for idx, shape in enumerate(shapes):
            pos_moves = possible_moves(occupied, shape)
            for pos_move in pos_moves:
                cur_weight, cur_params = weight(copy.deepcopy(occupied), shape, idx, pos_move, shapes) 
                if(cur_weight > top_weight):
                    top_weight = cur_weight
                    top_move = (shape, pos_move)
                    top_shape_idx = idx
                    top_params = cur_params
        return top_move, top_shape_idx, top_params

def get_random_init_shapes(n):
    random_shape_idx = np.random.choice(np.asarray(possible_shapes).shape[0], n, replace=True)
    print(random_shape_idx)
    return possible_shapes[random_shape_idx]



#TRAINING 

alpha = 0.0005
gamma = 0.9
epsilon =  0.5
weights = [1, 1, 1, -1,-1, -1, 1, 1]

for episode in range(5000):
    print('Episode: ' + str(episode))
    current_score = 0
    board, current_occupied = create_board()
    current_shapes = get_random_init_shapes(3)
    shape_idx_g = -1
    start = False
    while(check_moves_left(copy.deepcopy(current_occupied), current_shapes)):
        shapes_to_draw = copy.deepcopy(current_shapes)
        draw_board_with_shapes(board, current_occupied, shapes_to_draw)
        old_params = []
        if(start):
            old_params = get_params(copy.deepcopy(current_occupied), current_shapes, shape_idx)
            start = True
        else:
            old_params= [0,0,0,0,0,0,0,0]
        start = True
        best_move, shape_idx, new_params = get_best_move(copy.deepcopy(current_occupied), current_shapes)
        shape_idx_g = shape_idx
        current_occupied, row_lens = place_shape(copy.deepcopy(current_occupied), best_move[0], best_move[1])
        current_shapes[shape_idx] = get_random_init_shapes(1)
        if(epsilon > 0.01):
            epsilon *= 0.99
        R = -1
        if(len(row_lens) > 0):
            R = len(row_lens)*3
        check_moves_left_b = check_moves_left(copy.deepcopy(current_occupied), current_shapes)
        if(not check_moves_left_b):
            R = -10
        reg_term = 0
        if(not check_moves_left_b):
            with open("train_test.txt", "a") as file:
                file.write(str(weights) + ' - ' + str(current_score) + '\n')
            break
        for i in range(len(weights)):  
            weights[i] = weights[i] + alpha * weights[i] * (R - old_params[i] + gamma * new_params[i])
        #raw_available_shapes(board, current_shapes)
        print(weights)
        print(current_score)
        print(old_params)
        print(new_params)
        #print(new_params)
    

### PLAY ###
for episode in range(1000):
    print('Episode: ' + str(episode))
    current_score = 0
    board, current_occupied = create_board()
    current_shapes = get_random_init_shapes(3)
    shape_idx = -1
    start = False
    while(check_moves_left(copy.deepcopy(current_occupied), current_shapes)):
        shapes_to_draw = copy.deepcopy(current_shapes)
        old_params = []
        if(start):
            old_params = get_params(copy.deepcopy(current_occupied), shape_idx)
            start = True
        else:
            old_params= [0,0,0,0,0,0,0,0]
        draw_board_with_shapes(board, current_occupied, shapes_to_draw)
        best_move, shape_idx, new_params = get_best_move(copy.deepcopy(current_occupied), current_shapes)
        occupied_without_r = place_shape_search(copy.deepcopy(current_occupied), best_move[0], best_move[1])
        current_occupied, row_lens = place_shape(copy.deepcopy(current_occupied), best_move[0], best_move[1])
        current_shapes[shape_idx] = get_random_init_shapes(1)
        draw_board_with_shapes(board, occupied_without_r, current_shapes)
        check_moves_left_b = check_moves_left(copy.deepcopy(current_occupied), current_shapes)
        if(not check_moves_left_b):
            with open("scores.txt", "a") as file:
                file.write(str(weights) + ' - ' + str(current_score) + '\n')
            break
        print(current_score)
        print(old_params)
        print(new_params)