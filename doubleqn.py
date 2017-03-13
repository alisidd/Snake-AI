import os
import random
import math
from copy import deepcopy

import numpy as np
import tensorflow as tf

from constants import MOVES

class Move(object):
    up    = MOVES[3] #( 0,-1)
    down  = MOVES[1] #( 0, 1)
    left  = MOVES[2] #(-1, 0)
    right = MOVES[0] #( 1, 0)

VALID_ACTIONS = [Move.up, Move.down, Move.left, Move.right]

def manhattan_distance(head, other_snake):
    # Subtract the x and y coordinates and add them together to get the distance from head to other_snake
    return abs(head[0] - other_snake[0]) + abs(head[1] - other_snake[1])

class DoubleQNAgent:
    def __init__(self, strategies):
        # Game specific parameters
        self.snake_id = len(strategies)
        self.strategy = lambda id,s : self.get_action(s)
        self.grid_size = 20

        # Training parameters
        self.training = False
        load_model = True

        # Hyperparameters
        self.sight_distance = 13
        self.discount_factor = 0.99
        self.epsilon = 0.7
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.001
        self.batch_size = 32
        self.tau = 0.001

        # Setting up the features
        self.vision_grid_size = int(math.ceil(math.pow(self.sight_distance*2 + 1, 2) / 2))
        self.num_features = 5 * self.vision_grid_size + 1
        self.health = 100

        self.radars_index = {
            'self'      : 0,
            'enemy-head': 1 * self.vision_grid_size,
            'enemy-tail': 2 * self.vision_grid_size,
            'food'      : 3 * self.vision_grid_size,
            'wall'      : 4 * self.vision_grid_size
        }

        self.displacement = {}
        i = 0
        for x in xrange(-self.sight_distance, self.sight_distance + 1):
            for y in xrange(-self.sight_distance, self.sight_distance + 1):
                if manhattan_distance([0, 0], [x, y]) <= self.sight_distance:
                    self.displacement[(x, y)] = i
                    i += 1

        # Initializing the networks
        self.mainQN = DoubleQNetwork(self.num_features)
        self.targetQN = DoubleQNetwork(self.num_features)

        self.episode_rewards = []
        self.episode_mean_values = []
        self.episode_reward = 0
        self.episode_count = 0
        self.summary_writer = tf.summary.FileWriter("model/")

        self.saver = tf.train.Saver()

        trainables = tf.trainable_variables()
        self.targetOps = self.update_target_graph(trainables)
        self.experience_buffer = ExperienceBuffer()

        # Running the session
        self.session = tf.Session()

        self.session.run(tf.global_variables_initializer())
        if load_model:
            print("Restoring checkpoint...")
            self.saver.restore(self.session, 'model/data.chk')
        self.update_target(self.targetOps)
        print(self.session.run(self.mainQN.weights['h1']))
        print(self.session.run(self.mainQN.weights['out']))
        print(self.session.run(self.targetQN.weights['h1']))
        print(self.session.run(self.targetQN.weights['out']))

    def update_target_graph(self, tf_vars):
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars / 2]):
            op_holder.append(tf_vars[idx + total_vars/2].assign((var.value() * self.tau) + ((1-self.tau) * tf_vars[idx + total_vars/2].value())))
        return op_holder

    def update_target(self, op_holder):
        for op in op_holder:
            self.session.run(op)

    def get_action(self, state):
        self.radars = np.zeros(self.num_features)
        self.features = np.reshape(self.radars, (-1, self.num_features))

        if self.health < 15 and not self.is_eating(state):
            self.features[0, self.num_features - 1] = 1
        else:
            self.features[0, self.num_features - 1] = 0

        self.populate_vision(state)

        a = self.train_network(state)
        self.previous_state = deepcopy(state)
        self.previous_features = deepcopy(self.features)
        self.previous_a = deepcopy(a)

        if int(a) == 0:
            print("Up")
        elif int(a) == 1:
            print("Down")
        elif int(a) == 2:
            print("Left")
        elif int(a) == 3:
            print("Right")

        return VALID_ACTIONS[int(a)]

    def train_network(self, state, reward=0):
        if self.training:
            self.saver.save(self.session, 'model/data.chk')

        is_done = (reward != 0)

        if not is_done and not self.is_eating(state):
            self.health -= 1
            if self.health < 15:
                reward = -1

        if self.is_eating(state) or is_done:
            self.health = 100

        print('Health: {0}'.format(self.health))
        print('Reward: {0}'.format(reward))
        print("\n")

        if not is_done:
            self.episode_reward += 1
        elif self.training:
            self.episode_count += 1
            self.episode_rewards.append(self.episode_reward)

            # Statistics
            mean_reward = np.mean(self.episode_rewards[-5:])

            summary = tf.Summary()
            summary.value.add(tag='Performance/Reward', simple_value=float(mean_reward))
            self.summary_writer.add_summary(summary, self.episode_count)

            self.summary_writer.flush()

            self.episode_reward = 0


        # Get current state properties
        maxQprime, Qprime = self.session.run([self.mainQN.predict, self.mainQN.Qout], feed_dict={self.mainQN.X: self.features})
        print(np.matrix(Qprime))

        # Epsilon greedy policy
        if np.random.rand(1) < self.epsilon and self.training:
            action_to_take = np.array([np.random.randint(0, 4)])[0]
            print("Taking random action!")
        else:
            action_to_take = maxQprime

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

        # If this isn't the first step, then add to experience buffer
        if hasattr(self, 'previous_a'):
            # Add experience
            self.experience_buffer.add(np.reshape(np.array([self.previous_features, self.previous_a, reward, self.features, is_done]),[1, 5]))

            # If there's enough experiences in the buffer, then train on them
            if len(self.experience_buffer.buffer) >= self.batch_size and self.training:
                for _ in xrange(0, self.batch_size):
                    train_batch = self.experience_buffer.sample(1)

                    '''
                    Q1 is the best action for next state predicted from main network
                    Q2 is all the actions for next state predicted from target network
                    DoubleQ is the value of Q1 from Q2
                    targetQ is the output from the neural network for the previous features improved by changing the Q using DoubleQ's value
                    end_multiplier ensures that if an action caused the episode to end, then its Q value is only affected by the reward and not doubleQ
                    '''

                    Q1 = self.session.run(self.mainQN.predict, feed_dict={self.mainQN.X: np.vstack(train_batch[:,3])})
                    Q2 = self.session.run(self.targetQN.Qout, feed_dict={self.targetQN.X: np.vstack(train_batch[:,3])})

                    doubleQ = Q2[range(1), Q1]

                    targetQ = self.session.run(self.mainQN.Qout, feed_dict={self.mainQN.X: np.vstack(train_batch[:,0])})

                    end_multiplier = 1 - train_batch[:,4]

                    targetQ[0, train_batch[:,1][0]] = train_batch[:,2] + (self.discount_factor * doubleQ) * end_multiplier

                    # Update the network with our target values
                    _, error = self.session.run([self.mainQN.train_step, self.mainQN.error], feed_dict={self.mainQN.X: np.vstack(train_batch[:,0]), self.mainQN.nextQ: targetQ})

                self.update_target(self.targetOps)

        return action_to_take

    def is_eating(self, state):
        snakes = state.snakes

        for (snake_id, snake) in snakes.iteritems():
            if snake_id == self.snake_id:
                self_head = snake.position[0]

        if hasattr(self, 'previous_state'):
            for position, value in self.previous_state.candies.iteritems():
                if self_head[0] == position[0] and self_head[1] == position[1]:
                    return True

        return False

    def populate_vision(self, state):
        snakes = state.snakes

        for (snake_id, snake) in snakes.iteritems():
            if snake_id == self.snake_id:
                self_head = snake.position[0]

        # Add self
        for (snake_id, snake) in snakes.iteritems():
            if snake_id == self.snake_id:
                for snake_cell in snake.position:
                    if manhattan_distance(self_head, snake_cell) <= self.sight_distance:
                        self.add_to_self_grid(self_head, snake_cell, 'self')

        # Add enemy snakes
        for (snake_id, snake) in snakes.iteritems():
            if snake_id != self.snake_id:
                if manhattan_distance(self_head, snake.position[0]) <= self.sight_distance:
                    self.add_to_self_grid(self_head, snake.position[0], 'enemy-head')

        # Add enemy snakes' tails
        for (snake_id, snake) in snakes.iteritems():
            if snake_id != self.snake_id:
                for snake_cell in snake.position:
                    if manhattan_distance(self_head, snake_cell) <= self.sight_distance and snake_cell != snake.position[0]:
                        self.add_to_self_grid(self_head, snake_cell, 'enemy-tail')

        # Add food
        for position, value in state.candies.iteritems():
            if manhattan_distance(self_head, position) <= self.sight_distance:
                self.add_to_self_grid(self_head, position, 'food')

        # Add upper wall
        for i in xrange(0, self.grid_size + 1):
            cell = [i, -1]
            if manhattan_distance(self_head, cell) <= self.sight_distance:
                self.add_to_self_grid(self_head, cell, 'wall')

        # Add right wall
        for i in xrange(0, self.grid_size + 1):
            cell = [self.grid_size, i]
            if manhattan_distance(self_head, cell) <= self.sight_distance:
                self.add_to_self_grid(self_head, cell, 'wall')

        # Add down wall
        for i in xrange(0, self.grid_size + 1):
            cell = [i, self.grid_size]
            if manhattan_distance(self_head, cell) <= self.sight_distance:
                self.add_to_self_grid(self_head, cell, 'wall')

        # Add left wall
        for i in xrange(0, self.grid_size + 1):
            cell = [-1, i]
            if manhattan_distance(self_head, cell) <= self.sight_distance:
                self.add_to_self_grid(self_head, cell, 'wall')

    def add_to_self_grid(self, self_head, cell_to_modify, radar_to_modify):
        # Get distance relative to head of snake
        x_distance = (cell_to_modify[0] - self_head[0])
        y_distance = (cell_to_modify[1] - self_head[1])
        cell_from_head = (x_distance, y_distance)

        self.radars[self.radars_index[radar_to_modify] + self.displacement[cell_from_head]] = 1

class ExperienceBuffer:
    def __init__(self, buffer_size=500000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)),[size, 5])

class DoubleQNetwork:
    def __init__(self, num_features):
        self.intialize_network(num_features)

    def intialize_network(self, num_features):
        # Neural Network Parameters
        num_nodes = 20
        num_outputs = len(VALID_ACTIONS)

        # Input layer
        self.X = tf.placeholder(tf.float32, [None, num_features])

        # Hidden layer
        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([num_features, num_nodes], stddev=0.1)),
            'out': tf.Variable(tf.truncated_normal([num_nodes, num_outputs], stddev=0.1))
        }

        self.biases = {
            'h1': tf.Variable(tf.constant(0.1, shape=[num_nodes])),
            'out': tf.Variable(tf.constant(0.1, shape=[num_outputs]))
        }

        layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['h1'])
        layer_1 = tf.nn.relu(layer_1)

        # Output layer
        self.Qout = tf.add(tf.matmul(layer_1, self.weights['out']), self.biases['out'])
        self.predict = tf.argmax(self.Qout, 1)

        # Used to teach the neural network
        self.nextQ = tf.placeholder(tf.float32, [1, num_outputs])

        # How to improve the neural network
        self.error = tf.reduce_mean(tf.square(self.Qout - self.nextQ))
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.train_step = trainer.minimize(self.error)
