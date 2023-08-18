from gym import Env
from gym import spaces
from gym.spaces import Discrete, Box
import random
import numpy as np
import json
import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
from matplotlib import pyplot as plt


class UAVCollisionEnv(Env):
    def __init__(self):
        super(UAVCollisionEnv, self).__init__()

        # Load the JSON data
        json_data = """
        [
          {
            "position": [10, 10],
            "size": [10, 5]
          }

        ]
        """
        self.obstacles = json.loads(json_data)

        # Define the action and observation space
        self.action_space = spaces.Discrete(8)  # 8 actions for a UAV
        self.observation_space_grid = Box(low=0, high=0, shape=(5, 5),
                                          dtype=np.uint8)  # square  box of 5x5 around the uav of interest

        self.observaion_space_pos = Box(low=0, high=0, shape=(2,),
                                        dtype=np.float32)  # relative coordinate from uav of interest to destination

        self.observation_space = spaces.Tuple(
            (self.observaion_space_pos, self.observation_space_grid))  # complete observation space

        # Initialize the state
        self.state = np.array(
            [[2, 5], [15, 18], [19, 18], [5, 7], [10, 6], [10, 2], [5, 18], [5, 12], [2, 15], [15, 5], [15, 1], [15, 8],
             [7, 3], [9, 10], [9, 19], [17, 5]])  # Starting state
        self.tar_pos = self.state[15]
        self.UOI = self.state[0]

        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN

        self.exploration_prob = 1.0
        self.exploration_prob_decay = .005
        self.lr = .0001
        self.gamma = .95

        # We define our memory buffer where we will store our experiences
        # We stores only the 500000 last time steps

        self.memory_buffer = list()
        self.max_memory_buffer = 500000
        self.batch_size = 400
        self.state_size = 27
        self.action_size = 8

        # We creaate our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space
        self.model = Sequential([
            Dense(units=40, input_dim=self.state_size, activation='relu'),
            Dense(units=40, activation='relu'),
            Dense(units=self.action_size, activation='relu')
        ])
        self.model.compile(loss="mse",
                           optimizer=Adam(learning_rate=self.lr))

    def compute_action(self, current_state):
        # current_state=self.get_state()

        if np.random.uniform(0, 1) < self.exploration_prob:
            # return np.random.choice(range(self.action_space.sample()))
            return self.action_space.sample()

        #         q_values=self.model.predict(current_state)[o]
        q_values = self.model.predict(current_state)[0]

        return np.argmax(q_values)

    # At each time step, we store the corresponding experience
    def store_episode(self, current_state, action, reward, next_state, done):
        # We use a dictionnary to store them
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    def update_exp_prob(self):

        self.exploration_prob = self.exploration_prob * np.exp(- self.exploration_prob_decay)

    def cell_contains_object(self, position):
        #     """
        #     Checks if a cell at the given position contains an object.

        #     Args:
        #         position (tuple): The (x, y) position of the cell.

        #     Returns:
        #         bool: True if the cell contains an object, False otherwise.
        #     """
        x, y = position

        # Check if the cell collides with any obstacle
        for obstacle in self.obstacles:
            position = obstacle['position']
            size = obstacle['size']
            #         x_min = obstacle_position[0]
            #         y_min = obstacle_position[1]
            #         x_max = x_min + obstacle_size[0]
            #         y_max = y_min + obstacle_size[1]
            limit1 = size[0] + position[0]
            limit2 = size[1] + position[1]
            if position[0] <= x <= limit1 and position[1] <= y <= limit2:
                return True
                break
        #         else:
        #             return False

        #       if x_min <= x < x_max and y_min <= y < y_max:
        #             return True
        # Check if the cell collides with any mobile object (e.g., other UAVs)
        for uav_position in self.state[1:]:
            uav_x, uav_y = uav_position
            if uav_x == x and uav_y == y:
                return True

        return False

    def calculate_grid_square(self, agent_position, grid_size=5, area_size=(20, 20)):
        #
        #     Calculates the grid square surrounding the agent's position.

        #     Args:
        #         agent_position (tuple): Current position of the agent (x, y).
        #         grid_size (int): Size of each tile of the grid square.
        #         area_size (tuple): Dimensions of the area (X, Y).

        #     Returns:
        #         list: Grid square with binary values indicating object presence (1) or absence (0).

        #
        x, y = agent_position
        X, Y = area_size
        L = grid_size

        # Calculate the top-left corner of the grid square
        top_left_x = x - (L // 2)
        top_left_y = y - (L // 2)

        # Calculate the bottom-right corner of the grid square
        bottom_right_x = top_left_x + L
        bottom_right_y = top_left_y + L

        # Initialize the grid square
        grid_square = []

        # Iterate over the cells of the grid square
        for i in range(top_left_x, bottom_right_x):
            for j in range(top_left_y, bottom_right_y):
                # Check if the cell is within the area bounds
                if 0 <= i < X and 0 <= j < Y:
                    # Add 1 if there is an object in the cell, otherwise add 0
                    grid_square.append(1 if self.cell_contains_object((i, j)) else 0)
                else:
                    # Add 0 for cells outside the area bounds
                    grid_square.append(0)
        grid_square = np.array(grid_square).reshape((5, 5))
        return grid_square

    def collides_with_obstacle(grid, x, y):
        #     """
        #     Checks if the given coordinates (x, y) in the grid collide with an obstacle.

        #     Args:
        #         grid (list): The grid representing the intruders' positions.
        #         x (int): The x-coordinate to check.
        #         y (int): The y-coordinate to check.

        #     Returns:
        #         bool: True if the coordinates collide with an obstacle, False otherwise.
        #     """
        return grid[x][y] == 1

    def reached_destination(xrel, yrel):
        #     """
        #     Checks if the agent has reached its destination.

        #     Args:
        #         xrel (float): The normalized x-coordinate of the relative position.
        #         yrel (float): The normalized y-coordinate of the relative position.

        #     Returns:
        #         bool: True if the agent has reached the destination, False otherwise.
        #     """
        return xrel == 0 and yrel == 0

    #     def create_grid(X, Y):
    #     """
    #     Creates an empty grid of dimensions X × Y.

    #     Args:
    #         X (int): The number of rows in the grid.
    #         Y (int): The number of columns in the grid.

    #     Returns:
    #         list: A 2D grid represented as a list of lists.
    #     """
    #      return [[0] * Y for _ in range(X)]

    def find_coordinates(grid, value):
        #     """
        #     Finds the coordinates of a given value in the grid.

        #     Args:
        #         grid (list): The grid to search for the value.
        #         value: The value to find in the grid.

        #     Returns:
        #         list: A list of coordinates where the value is found.
        #     """
        coordinates = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == value:
                    coordinates.append((i, j))
        return coordinates

    def step(self, action):
        current_state = self.state[0]
        current_x, current_y = current_state

        # Define the directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        direction = directions[action]
        new_x = current_x + direction[0]
        new_y = current_y + direction[1]

        # Compute the new state and distance vector Prel
        new_state = np.array([new_x, new_y])
        distance_vector = (self.tar_pos - new_state) / np.array([100, 100])  # X,Y= 100,100

        # Check for collisions and calculate the reward
        done = False
        reward = -0.1  # Penalty for each step

        if self.cell_contains_object((new_x, new_y)):
            done = True
            reward = -100
        elif distance_vector[0] == 0 and distance_vector[1] == 0:
            done = True
            reward = 100

        # Store the episode
        #      self.store_episode(current_state, action, reward, new_state, done)

        # Update the environment state
        #      self.state[0] = new_state
        # Concatenate the relative position and grid square
        current_grid_square = self.calculate_grid_square(new_state)
        current_state = np.concatenate((distance_vector, current_grid_square.flatten()))

        #      return (distance_vector, self.calculate_grid_square(new_state), reward, done, {})
        return (current_state, reward, done, {})

    #     def reset(self):
    #         # Reset the environment state
    # #         self.state = self.state[0]
    #         self.UOI=self.state[0]
    #         return self.UOI
    def reset(self):
        # Reset the environment state
        #      self.state[0] = np.array([10, 50])  # Reset the position of the UAV of interest
        self.UOI = self.state[0]
        self.UOI = np.array(self.UOI)
        distance_vector = (self.tar_pos - self.UOI) / np.array([100, 100])

        # Concatenate the relative position and grid square
        current_grid_square = self.calculate_grid_square(self.UOI)
        #      distance_vector=list(distance_vector)

        #      distance_vector.append(current_grid_square.flatten())
        current_state = np.concatenate((distance_vector, current_grid_square.flatten()))

        #      current_state = np.array(distance_vector,)
        #      return current_state.reshape((1, self.state_size))
        return current_state

    # At the end of each episode, we train our model

    # At the end of each episode, we train our model
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]

        # We iterate over the selected experiences
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma * np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            # train the model
            self.model.fit(experience["current_state"], q_current_state, verbose=0)

    def render(self):
        # Render the environment
        fig, ax = plt.subplots()

        # Plot the obstacles
        for obstacle in self.obstacles:
            position = obstacle['position']
            size = obstacle['size']
            rect = plt.Rectangle(position, size[0], size[1], color='black')
            ax.add_patch(rect)

        # Plot the UAVs
        for i in range(16):
            x, y = self.state[i]
            if i == 0:  # First UAV is the UAV of interest
                color = 'blue'  # Color for the UAV of interest
            else:
                color = 'red'  # Color for other UAVs
            circle = plt.Circle((x, y), 0.5, color=color)
            ax.add_patch(circle)

        # Plot the targeted position for the UAV of Interest
        target_x = 17
        target_y = 5
        target_circle = plt.Circle((target_x, target_y), 0.5, color='green', label='Target Position')
        ax.add_patch(target_circle)

        ax.set_xlim([0, 20])
        ax.set_ylim([0, 20])

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='UAV of Interest', markerfacecolor='blue', markersize=5),
            plt.Line2D([0], [0], marker='o', color='w', label='Other UAVs', markerfacecolor='red', markersize=5),
            plt.Rectangle((0, 0), 1, 1, color='black', label='Obstacle'),
            plt.Line2D([0], [0], marker='o', color='w', label='Targeted Position', markerfacecolor='green',
                       markersize=5)
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.2, 1))

        plt.show()


env=UAVCollisionEnv()

# Number of episodes to run
n_episodes = 1000
# Max iterations per epiode
max_iteration_ep =500
total_steps = 0
episode_rewards = []  # List to store cumulative rewards per episode

# We iterate over episodes
for e in range(n_episodes):
    # We initialize the first state and reshape it to fit
    #  with the input layer of the DNN
    current_state = env.reset()
    current_state = np.array([current_state])
    #     current_state = current_state.reshape((None, env.state_size))
    cumulative_reward = 0  # Initialize cumulative reward for the episode
    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        # the agent computes the action to perform
        action = env.compute_action(current_state)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        next_state, reward, done, a = env.step(action)
        next_state = np.array([next_state])
        #         next_state = next_state.reshape((None, env.state_size))
        cumulative_reward += reward

        # We sotre each experience in the memory buffer
        env.store_episode(current_state, action, reward, next_state, done)

        # if the episode is ended, we leave the loop after
        # updating the exploration probability
        if done:
            env.update_exp_prob()
            break
        current_state = next_state

    episode_rewards.append(cumulative_reward)  # Append cumulative reward to the list
    #     print(episode_rewards)
    # if the have at least batch_size experiences in the memory buffer
    # than we tain our model
    if total_steps >= env.batch_size:
        env.train()
# Plotting the episode rewards
plt.plot(range(1000), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Number of Episodes vs. Cumulative Reward')
plt.show()

import matplotlib.pyplot as plt

# Initialize collision count and episode count
collision_count = 0
episode_count = 0

# Create a list to store collision percentages
collision_percentages = []

# Iterate over the episode rewards
for reward in episode_rewards:
    episode_count += 1

    # Check if the reward is below a threshold (indicating a collision)
    if reward < -1:
        collision_count += 1

    # Calculate the collision percentage for the current episode
    collision_percentage = (collision_count / episode_count) * 100

    # Add the collision percentage to the list
    collision_percentages.append(collision_percentage)

# Plot the percentage of collision
plt.plot(range(1, episode_count + 1), collision_percentages)
plt.xlabel('Episode')
plt.ylabel('% of Collision')
plt.title('Percentage of Collision vs Episodes')
plt.show()
