This is an AI maze solver.
It uses reinforcement learning to get from the start point to the goal.

Specifically, it uses Q-learning and a Q-table to find the optimal route to take.
Every occupiable state on the maze grid gets a spot in the Q-table with a score for each direction of movement(up, down, left, and right).
The scores are constantly updated based on new info that the agent recieves.
The agent uses these scores to determine the best move to take from each state, until it reaches the goal.

The solver is preloaded with a 40x40 maze, but can be changed to any size. **Note that sizes bigger than 50x50 have not been tested yet, and may not work**

The maze needs to be in the format of nested arrays, where...
  "S" represents the starting point
  "G" represents the goal/ending point
  " " represents free areas(usable spaces)
  "#" represents walls(unusable spaces)

The agent will go through the maze 5000 times(episodes) and then will proceed to display the optimal route.
The display will show the provided maze as a grid, where A represents the agents current location on the maze, and * represents the path the agent took to get there.

You may run the Maze_Solver.py as-is to get the agent to solve the 40x40, or you can input another maze at Step 2 of the code. 
The "Mazes" folder has been loaded with 8 different sizes that you can use, or you may make your own!
