# Maze Puzzle

This puzzle was deviced by [Desmoswoman](https://discord.com/channels/655972529030037504/674475911445217291/1450629398729326685) in Discord. I coded this the way I understood her instructions.

## Da rules

### Goal
The goal of the game is to reach the goal in the maze with the fewest number of moves and do it multiple times with a specific policy (see below). Valid moves include 4 orthogonal directions, where some tiles may be obstacles. There are three type of obstacles:
1. **abyss**: not occupiable, but traversable under some circumstances (see traversal policy)
2. **abyss with a bridge**: not occupiable but always traversable
3. **abyss with a trap**: not occupiable and not traversable

### Traversal policy

The game considers a _policy_ to determine if the player is allowed to traverse (but not occupy) an abyss. The policies are the following

* `green tile | abyss | any tile`
* `any tile | abyss | purple tile`
* `any tile | bridge | any tile`

If none of the explicit conditions above are met and the abyss has no trap, then you can still traverse it. The policy counts the explicit conditions met which we will call `n`.

Valid routes to the end are when `n` equals 0, 1, 2, and 3 throughout the entire run. You must complete all 4 to win conquer the maze.

> Note: the original rules say that the obstacles are walls, but I think it is more natural to think of them as holes, so I will use the term "abyss" instead.

## Visualization

The solver uses curses to show an animation of the solution path.

The valid value of `n` can be passed as parameter to the function.

The mazes can be added in the `./puzzles` directory. Use the naming convention of `weird_maze_2*` and the program will automatically prompt it.
