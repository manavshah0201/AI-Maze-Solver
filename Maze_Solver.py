# Manav Shah
# 2025/11/30
# AI Maze Solver
# Has been successfully tested on maze sizes up to 40x40

# Step 1: Importing libraries

import numpy as np
import time
import random
from IPython.display import clear_output

# Step 2: Defining the maze layout
# The maze can be changed for difficulty
# "S" represents the start, "G" represents the goal or end, " " represents empty space(usable), and "#" represents a wall(unusable)
# Maze needs to be formatted with nested arrays to provide 2 dimensions

maze = [
    ['S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#'],
    [' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#'],
    [' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', '#', ' ', '#'],
    [' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#'],
    [' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#'],
    [' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#'],
    [' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#'],
    [' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#'],
    [' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#'],
    [' ', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#'],
    [' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#'],
    ['#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#'],
    [' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', '#'],
    [' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#'],
    [' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', '#'],
    [' ', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', ' ', '#'],
    [' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#'],
    ['#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#'],
    [' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#'],
    [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#'],
    [' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
    [' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#'],
    [' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#'],
    [' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#'],
    ['#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#'],
    [' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#'],
    [' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#'],
    [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#'],
    [' ', '#', ' ', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#'],
    [' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#'],
    [' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#'],
    [' ', '#', ' ', '#', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', '#'],
    [' ', '#', '#', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#'],
    [' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', 'G'],
]

# Step 3: Defining the actions that the agent can take

actions = ["up", "down", "left", "right"]

# Step 4: Initializing the Q-table for the maze to store rewards for each state-action pair

Q = {}
rows = len(maze)
cols = len(maze[0])

for a in range(rows):
  for b in range(cols):
    if maze[a][b] != "#":
      Q[(a, b)] = np.zeros(len(actions))
    if maze[a][b] == "S":
      start = (a,b)
    elif maze[a][b] == "G":
      goal = (a,b)

# Step 5: Defining the function to take the action and give a reward for the action

def take_action(state, action):
  i, j = state

  if action == "up":
    i -= 1

  if action == "down":
    i += 1

  if action == "left":
    j -= 1

  if action == "right":
    j += 1

  # Checks if action causes agent to hit boundaries or walls
  if i < 0 or i >= rows or j < 0 or j >= cols or maze[i][j] == "#":
    return state, -15   # penalty for hitting wall

  elif maze[i][j] == "G":
    return (i,j), 50   # reward for reaching the end

  else:
    return (i,j), -5   #small penalty to find the shortest route

# Step 6: Defining parameters that guide the learning of the agent

epsilon = 0.8     # epsilon value determines whether the agent explores or exploits
decay_rate = 0.995 # rate of decrease for epsilon value
min_epsilon = 0.2 # lowest value that epsilon can go to

alpha = 0.2       # alpha is the learning rate
gamma = 1       # gamma is the discount factor

episodes = 5000   # the amount of times the agent runs through the maze to learn

# Step 7: Using epsilon greedy action selection to determine next action

# using the epsilon value to define the next action
def choose_action(state):

  if random.uniform(0,1) < epsilon:
    # the agent will explore(pick a random action)
    return random.choice(range(len(actions)))

  else:
    # the agent will exploit(pick the action with the best recorded output)
    return np.argmax(Q[state])

# Step 8: Defining the function that updates the Q-table with

def update_Q(state, action_index, reward, next_state):
  best_next = np.max(Q[next_state])
  Q[state][action_index] += alpha * (reward + gamma * best_next - Q[state][action_index])

# Step 9: Defining the function that shows the agents position in the maze

def display_maze(agent_pos):

    row, col = agent_pos
    maze[row][col] = f"\033[91m*\033[0m"

    for r in range(rows):
        row_display = ""
        for c in range(cols):
            if (r, c) == agent_pos:
                row_display += "A "   # Agent
            else:
                row_display += maze[r][c] + " "
        print(row_display)
    print()

# Step 10: Defines the function that shows the final result of its learning (The shortest path it could find)

def run_agent():
    state = start
    steps = 0

    while state != goal:

        # clear the screen for next move
        clear_output(wait=True)

        # show maze
        display_maze(state)

        # choose best learned action (NO exploration)
        action_index = np.argmax(Q[state])
        action = actions[action_index]

        # move
        next_state, reward = take_action(state, action)
        state = next_state

        time.sleep(0.1)  # slow down so you can see movement
        steps += 1

    if state == goal:

      # Clears screen
      clear_output(wait=True)

      # show maze
      display_maze(state)
      print("Goal reached in", steps, "steps!")

# Step 11: Defines the function to show the final Q-table

def show_Q_table():
  print("Final Q-table:\n")
  for state in sorted(Q.keys()):
    print(f"{state}: {Q[state]}")

# Step 12: Training the agent through repitition

for ep in range(episodes):
  state = start
  moves = 0

  while state != goal:

    if moves > 50000:
      print("reached 50000 moves")
      break
    # choosing an action
    action_index = choose_action(state)
    action = actions[action_index]

    # take the action
    next_state, reward = take_action(state, action)

    # update the Q-table
    update_Q(state, action_index, reward, next_state)

    # move the agent to next state
    state = next_state

    # increase variable for number of moves that have been taken
    moves += 1

    # lowering epsilon value
    if ep > 1000:
      epsilon = max(min_epsilon, epsilon * decay_rate)

  if state == goal:
    if (ep+1) % 20 == 0:
      clear_output(wait=True)
      print(f"Episode {ep+1}/{episodes} completed")

# Step 13: Runs the function to show final result

run_agent()

# Step 14: Runs the function to show the final Q-table

show_Q_table()
