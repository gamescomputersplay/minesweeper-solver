# Minesweeper solver / bot

### N-dimensional Solver / simulator 

Minesweeper solver simulates or plays minesweeper games. It uses both deterministic and probability-based logic to come up with its moves, reaching the win rate of 38.6±0.3% for a standard Expert game (30x16, 99 mines).

Simulator also supports N-dimensional minesweeper games. With N from 1 to anything.

Note, that although there is no upper boundary on the number of dimensions, it's the computational explosion that provides a limit. 7-dimensional 4^7 slows to a crawl (it requires several minutes per one game simulation), and 8-dimensional game crashed, depleting all my laptop's memory in the process.

It also supports wrapping the field around on itself. In case of 1D that's a circumference of a circle, for 2D that would correspond to the surface of a 3D torus. For N dimensions - surface of N+1 torus, I guess???

### Bot 

Bot is an interface to the solver that would take screenshots, read the data, feed it to the solver and click the cells. It supports:

- Minesweeper clones like Minesweeper X / Vienna / Arbiter.
- A 4D version of minesweeper (search for "4D Minesweeper" on Steam).

### Extending functionalities

Solver can be used to find the best move in a given position, although there is no user-friendly interface for that. Basically you need to initiate a new Minesweeper Solver object with the right field size and mine count, then use the solve method, giving it a NumPy array with cells known so far.

There is a way to add support to other minesweeper variations (say, Google Minesweeper), although there is no user-friendly interface for that. Basically, you need to create a new MinesweeperSettings object, setting colors of the closed cells (this is how bot finds the game on the screen) and samples of all numbers it would encounter.

## Files

### minesweeper_game.py
Minesweeper game itself, different game presets, plus some helper functions and constants, used throughout the project.

### minesweeper_solver.py
The main part of the solver. Solver is stateless - all you need to do is make it aware of the game parameters (board size, and total number of mines) and pass in the current board. It will give you two lists: cells that are safe to click (no mines), and cells with mines. It also returns some stats about methods used to arrive at this conclusion.

### minesweeper_classes.py
Some classes used by the solver. Actually most of the heavy lifting is happening here, not in the solver file.

### minesweeper_sim.py
Functions for simulating multiple minesweeper games, together with some tools to aggregate and show statistics about the simulation.

### minesweeper_bot.py
The bot: would take a screenshot, find a minesweeper game on it, and proceed to playing it multiple times, keeping track of the number of wins. Currently works with Arbiter / Vienna / Minesweeper X with standard skins. Also works with 4D Minesweeper available on Steam.

## Solution methods

In an attempt to find the best cells to click (or marked as mines), solver goes through a number of methods - from trivial ones to more complex. As long as a method yields results - solver stops and returns that result. Here are the methods (Percentage is how many cells are solved using this method in an average 2D expert game).

Program starts with generating groups - a simplest amount of information about mines, as in "cells A, B, C have X remaining mines". Next few methods are based on those groups.

### First click (0.5%)

First click is always on an all-zero coordinate cell.

### Naive (88%)

If there are zero mines in a group - all cells are safe. If there are as many mines as cells - all cells are mines.

### Groups (8%)

Search for groups that are included in another group and deduce safe/mines from that. For example:
- cells AB have 1 mine; cells ABC have 1 mine: cell C is safe
- cells AB have 1 mine, cells ABC have 2 mines: cell C is mine

### Subgroups (1.4%)

Break down existing groups into "no more than" and "at least" subgroups. For example:
- if ABCD, have 2 mines, then: ABC has at least than 1 mine, Same for ABD, BCD and so on
- if ABC has 1 mine, then: AB, BC CA have no more than 1 mine.
Cross check three subgroups with regular groups and deduce safe cells and mines.

### Coverage (0.2%)

By coverage here I mean "how many mines are covered by the information I have". For example if I have a 50/50, I don't know where the mine is, but I know that those two cells contain exactly 1 mine.

The solver tries to attribute as many mines as possible to known groups. Ideally, why try to know how many mines are exactly in the groups. Calculate number of mines in remaining "uncovered" cells. If it is zero - they all are safe. If there are as many mines as cells - all are mines.

### CSP (0.2%)

Stands for Constraint Satisfaction Problem

Program isolates an area of interconnected groups - in the program it is called a Cluster. (For example if you started solving from two opposite ends and opened areas that haven't "met" yet - you have 2 clusters).

For each cluster program bruteforce all possible solutions. If a cell is safe in all solutions - it is a safe cell. If it is always a mine - it is a mine.

### Bruteforce (0.02%)

Kicks in whenever there are few enough remaining cells and mines. Same as CSP, but for all cells. Serves as a last resort in case previous methods missed something, but looking at how rarely it yields results, previous methods are pretty good.

### Probability (1.7%)

When all deterministic methods listed above fail, the program tries to find the best cell to click, based on mine probabilities and a few other indicators. Here they are:

- chance that the cell has a mine
- chance that the cell has a zero (will trigger opening of neighbor cells)
- is close to "frontier" (covered cells that are next to any numbers)
- how many safe guaranteed safe cells will there be next time (according to CSP/Bruteforce results)

For top potential cells the program calculates what number may appear in this cell if clicked on, and tries to find a solution for the next move too. This way it calculates a few additional factors:

- how many deterministic cells will be on average for the next move
- chance there will be any deterministic solution for the next move
- survival chance after next move (depends on the lowest mine chance in this move and the next one)


This minesweeper solver / bot is written in Python as a hobby project by GamesComputerPlay, check out [my youtube channel](https://www.youtube.com/c/GamesComputersPlay) for other "Python for games" stuff.

