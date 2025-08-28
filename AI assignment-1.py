#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from collections import deque
import heapq
import tracemalloc

# Defining maze
maze = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]])

start = (0, 1)
end = (9, 18)


# Get neighbours function
def get_neighbours(a):
    neighbours = []
    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    for dx, dy in directions:
        nx, ny = a[0] + dx, a[1] + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
            neighbours.append((nx, ny))
    return neighbours


# BFS Algorithm
def BFS(start, end):
    queue = deque([start])
    visited = [start]
    relations = {}

    while queue:
        current = queue.popleft()
        if current == end:
            path = []
            while current in relations:
                path.append(current)
                current = relations[current]
            path.append(start)
            return path[::-1]

        neighbours = get_neighbours(current)
        for neighbour in neighbours:
            if neighbour not in visited:
                queue.append(neighbour)
                visited.append(neighbour)
                relations[neighbour] = current

    return None


# DFS Algorithm
def DFS(start, end):
    stack = deque([start])
    visited = [start]
    relations = {}

    while stack:
        current = stack.pop()
        if current == end:
            path = []
            while current in relations:
                path.append(current)
                current = relations[current]
            path.append(start)
            return path[::-1]

        neighbours = get_neighbours(current)
        for neighbour in neighbours:
            if neighbour not in visited:
                stack.append(neighbour)
                visited.append(neighbour)
                relations[neighbour] = current

    return None


# A* Algorithm
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def A_star(start, end):
    heap = []
    relations = {}
    g_cost = {start: 0}
    heapq.heappush(heap, (manhattan_distance(start, end), start))

    while heap:
        _, current = heapq.heappop(heap)
        if current == end:
            path = []
            while current in relations:
                path.append(current)
                current = relations[current]
            path.append(start)
            return path[::-1]

        neighbours = get_neighbours(current)
        for neighbour in neighbours:
            if neighbour in relations:
                continue
            new_g_cost = g_cost[current] + 1
            if neighbour not in g_cost or new_g_cost < g_cost[neighbour]:
                g_cost[neighbour] = new_g_cost
                f_cost = new_g_cost + manhattan_distance(neighbour, end)
                heapq.heappush(heap, (f_cost, neighbour))
                relations[neighbour] = current

    return None


# Visualize the maze with the path
def visualize_path(path):
    if path is None:
        print("No path found")
        return
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if (i, j) == start:
                print("S", end=" ")
            elif (i, j) == end:
                print("E", end=" ")
            elif (i, j) in path:
                print("*", end=" ")
            else:
                if maze[i, j] == 1:
                    print("1", end=" ")
                else:
                    print("0", end=" ")
        print("\n")


# Main program execution
graphSearchAlgorithms = {'BFS': BFS, 'DFS': DFS, 'A*': A_star}

for algorithm_name, algorithm_func in graphSearchAlgorithms.items():
    print(f"\n{algorithm_name}\n")

    # Start memory tracking
    tracemalloc.start()

    # Get path
    path = algorithm_func(start, end)

    # Stop memory tracking
    memory_used = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Visualize the path
    visualize_path(path)

    if path:
        print(f"Path: {path}")
        print(f"Path length: {len(path)}")
    print(f"Memory used: {memory_used}")
