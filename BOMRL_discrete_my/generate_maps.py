#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:24:05 2023

@author: robotics
"""
import numpy as np
def generate_random_map(size=4, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    res=[]
    while not valid:
        p = min(1, p)
        res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res[0][0] = "S"
        res[-1][-1] = "G"
        for i in range(size):
            for j in range(size):
                if i+j<2.5:
                    res[i][j] = "F"
        res[0][0] = "S"
        res[-1][-1] = "G"
        valid = is_valid(res)
    
    return ["".join(x) for x in res]


size = 4
def is_valid(res):
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                    continue
                if res[r_new][c_new] == "G":
                    return True
                if res[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


m=None
for i in range(10):
    m=generate_random_map(size=4, p=0.8)
    print(m)
    with open('maps/map'+str(i)+'.npy', 'wb') as f:
        np.save(f,m)

