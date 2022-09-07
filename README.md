# Minesweeper solver / bot

Minesweeper solver simulates or plays minesweeper games. It uses both deterministic and probability-based logic to come up with its moves, reaching the win rate of 37.1% for a standard Expert game (30x16, 99 mines). Simulator also supports N-dimensional minesweeper games.

The bot can play classical 2D minesweeper by taking screenshots and clicking cells it finds on the screen - currently major minesweeper clones like Minesweeper X and Arbiter are supported.

This minesweeper solver / bot is written in Python as a hobby project by GamesComputerPlay, check out [my youtube channel](https://www.youtube.com/c/GamesComputersPlay) for other "Python for games" stuff.

## Files

### minesweeper_game.py
Minesweeper game itself itself, plus some helper functions and constants, used throughout the project

### minesweeper_solver.py
The main part of teh solver. Solver is stateless - all you need to do is make it aware of the game parameters (board size, and total number of mines) and pass in the current board. It will give you two lists: cells that are safe to click (no mines), and cells with mines. It also returns some stats about methods were used to arrive to this conclusion.

### minesweeper_classes.py
Some classes used by solver. Actually most of heavy lifting happening here, not in the solver file.

### minesweeper_sim.py
Functions for simulating multiple minesweeper games, together with some tools to aggregate and show statistics about the simulation.

### minesweeper_bot.py
The bot: would take a screenshot, find a minesweeper game on it, and proceed to playing it multiple times, keeping track of the number of wins. Currently works with Arbiter and Minesweeper X with standard skins.
