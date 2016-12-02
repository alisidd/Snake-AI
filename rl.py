"""
Reinforcement learning
"""

import random, math, pickle
import interface, utils
from collections import defaultdict
from utils import progressBar
from copy import deepcopy

class QLearningAlgorithm:
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2, weights = None):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.numIters = 0
        
        if weights:
            with open(weights, "rb") as fin:
                weights_ = pickle.load(fin)
                self.weights = weights_
        else:
            self.weights = defaultdict(float)


    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        """
        The strategy implemented by this algorithm.
        With probability `explorationProb` take a random action.
        """
        self.numIters += 1
        if len(self.actions(state)) == 0:
            return None
        
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.evalQ(state, action), action) for action in self.actions(state))[1]

    def getStepSize(self):
        """
        Get the step size to update the weights.
        """
        return 1.0 / math.sqrt(self.numIters)

    def incorporateFeedback(self, state, action, reward, newState):
        if newState is None:
            return
        
        phi = self.featureExtractor(state, action)
        pred = sum(self.weights[k] * v for k,v in phi)
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions(newState))
        except:
            v_opt = 0.
        target = reward + self.discount * v_opt
        for k,v in phi:
            self.weights[k] = self.weights[k] - self.getStepSize() * (pred - target) * v

    def train(self, strategies, grid_size, num_trials=100, max_iter=1000, verbose=False):
        print "RL training"
        totalRewards = []  # The rewards we get on each trial
        rl_id = len(strategies)
        for trial in xrange(num_trials):
            progressBar(trial, num_trials)
            game = interface.Game(grid_size, len(strategies) + 1, candy_ratio = 1., max_iter = max_iter)
            state = game.startState()
            totalDiscount = 1
            totalReward = 0
            points = state.snakes[rl_id].points
            while not game.isEnd(state) and rl_id in state.snakes:
                # Compute the actions for each player following its strategy
                actions = {i: strategies[i](i, state) for i in state.snakes.keys() if i != rl_id}
                action = self.getAction(state)
                actions[rl_id] = action

                newState = game.succ(state, actions)
                if rl_id in newState.snakes:
                    if len(newState.snakes) == 1: # it won
                        reward = 2.0 * newState.snakes[rl_id].points
                    else:
                        reward = newState.snakes[rl_id].points - points

                    points = newState.snakes[rl_id].points
                    self.incorporateFeedback(state, action, reward, newState)
                else: # it died
                    reward = - points
                    self.incorporateFeedback(state, action, reward, newState)

                totalReward += totalDiscount * reward
                totalDiscount *= self.discount
                state = newState

            if verbose:
                print "Trial %d (totalReward = %s)" % (trial, totalReward)
            totalRewards.append(totalReward)

        progressBar(num_trials, num_trials)
        print "Average reward:", sum(totalRewards)/num_trials
        return totalRewards


class QLambdaLearningAlgorithm:
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2, lambda_ = 0.2, weights = None):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.numIters = 0
        self.lambda_ = lambda_

        if weights:
            with open(weights, "rb") as fin:
                weights_ = pickle.load(fin)
                self.weights = weights_
        else:
            self.weights = defaultdict(float)

    def getAction(self, state):
        """
        The strategy implemented by this algorithm.
        With probability `explorationProb` take a random action.
        Return `action, is_optimal_action`.
        """
        self.numIters += 1
        if len(self.actions(state)) == 0:
            return None
        
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state)), False
        else:
            return max((self.evalQ(state, action), action) for action in self.actions(state))[1], True

    def incorporateFeedback(self, state, action, reward, newState, history = []):
        if newState is None:
            return
        
        phi = self.featureExtractor(state, action)

        pred = sum(self.weights[k] * v for k,v in phi)
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions(newState))
        except:
            v_opt = 0.
        target = reward + self.discount * v_opt
        delta = pred - target
        for k,v in phi:
            self.weights[k] = self.weights[k] - self.getStepSize() * delta * v

        for i in xrange(len(history)):
            s, a = history[-(i+1)] 
            delta *= self.discount * self.lambda_
            self.incorporateFeedback(s, a, delta)

    def incorporateFeedback(self, state, action, delta):
        phi = self.featureExtractor(state, action)
        for k,v in phi:
            self.weights[k] = self.weights[k] - self.getStepSize() * delta * v

    def train(self, strategies, grid_size, num_trials = 100, max_iter = 1000, verbose = False):
        print "RL training"
        totalRewards = []  # The rewards we get on each trial
        rl_id = len(strategies)
        for trial in xrange(num_trials):
            progressBar(trial, num_trials)
            game = interface.Game(grid_size, len(strategies) + 1, candy_ratio = 1., max_iter = max_iter)
            state = game.startState()
            totalDiscount = 1
            totalReward = 0
            points = state.snakes[rl_id].points
            history = []
            while not game.isEnd(state) and rl_id in state.snakes:
                # Compute the actions for each player following its strategy
                actions = {i: strategies[i](i, state) for i in state.snakes.keys() if i != rl_id}
                action, optimal_action = self.getAction(state)
                actions[rl_id] = action

                newState = game.succ(state, actions)
                if rl_id in newState.snakes:
                    if len(newState.snakes) == 1: # it won
                        reward = 2.0 * newState.snakes[rl_id].points
                    else:
                        reward = newState.snakes[rl_id].points - points

                    points = newState.snakes[rl_id].points
                    self.incorporateFeedback(state, action, reward, newState, history)
                else: # it died
                    reward = - points
                    self.incorporateFeedback(state, action, reward, newState, history)

                # add decsion to history, or reset if non-greedy choice
                if optimal_action:
                    history.append((state, action))
                else:
                    history = []

                totalReward += totalDiscount * reward
                totalDiscount *= self.discount
                state = newState

            if verbose:
                print "Trial %d (totalReward = %s)" % (trial, totalReward)
            totalRewards.append(totalReward)

        progressBar(num_trials, num_trials)
        print "Average reward:", sum(totalRewards)/num_trials
        return totalRewards

############################################################

def simpleFeatureExtractor0(state, action, id):
    head = state.snakes[id].position[0]
    features = [(('candy', utils.add(head, c, mu = -1), v, action.dir, action.n), 1.) for c,v in state.candies.iteritems()]
    features += [(('adv', utils.add(head, s.position[0], mu = -1), action.dir, action.n), 1.) for k,s in state.snakes.iteritems() if k != id]
    features += [(('x', head[0], action.dir, action.n), 1.), (('y', head[1], action.dir, action.n), 1.)]
    return features

def simpleFeatureExtractor1(state, action, id):
    if action is None:
        dir_ = None
        norm_ = None
    else:
        dir_ = action.direction()
        norm_ = action.norm()

    head = state.snakes[id].position[0]
    features = [(('candy', utils.add(head, c, mu = -1), dir_, norm_), 1.) for c,v in state.candies.iteritems() if utils.dist(head, c) < 12]
    features += [(('adv', utils.add(head, t, mu = -1), dir_, norm_), 1.) for k,s in state.snakes.iteritems() for t in s.position if k != id and utils.dist(head, t) < 12]
    features += [(('my-tail', utils.add(head, t, mu = -1), dir_, norm_), 1.) for t in state.snakes[id].position[1:] if utils.dist(head, t) < 12]
    features += [(('x', head[0], dir_, norm_), 1.), (('y', head[1], dir_, norm_), 1.)]
    return features

def simpleFeatureExtractor2(state, action, id):
    head = state.snakes[id].position[0]
    features = [(('candy', utils.add(head, c, mu = -1), v, action.dir, action.n), 1.) for c,v in state.candies.iteritems()]
    features += [(('adv-head', utils.add(head, s.position[0], mu = -1), action.dir, action.n), 1.) for k,s in state.snakes.iteritems() if k != id and utils.dist(head, s.position[0]) < 7]
    features += [(('adv-tail', utils.add(head, t, mu = -1), action.dir, action.n), 1.) for k,s in state.snakes.iteritems() for t in s.position[1:] if k != id and utils.dist(head, t) < 7]
    features += [(('my-tail', utils.add(head, t, mu = -1), action.dir, action.n), 1.) for t in state.snakes[id].position[1:] if utils.dist(head, t) < 7]
    features += [(('x', head[0], action.dir, action.n), 1.), (('y', head[1], action.dir, action.n), 1.)]
    return features

############################################################



def rl_strategy(strategies, featureExtractor, grid_size, num_trials=100, max_iter=1000, filename = "weights.p", verbose=False):
    rl_id = len(strategies)
    actions = lambda s : s.simple_actions(rl_id)
    features = lambda s,a : featureExtractor(s, a, rl_id)
    rl = QLearningAlgorithm(actions, discount = .9, featureExtractor = features, explorationProb = 0.3)
    rl.train(strategies, grid_size, num_trials=num_trials, max_iter=max_iter, verbose=verbose)
    rl.explorationProb = 0
    strategy = lambda id,s : rl.getAction(s)

    # save learned weights
    with open(filename, "wb") as fout:
        weights = dict(rl.weights)
        pickle.dump(weights, fout)
    
    return strategy

def load_rl_strategy(filename, strategies, featureExtractor):
    rl_id = len(strategies)
    actions = lambda s : s.simple_actions(rl_id)
    features = lambda s,a : featureExtractor(s, a, rl_id)
    rl = QLearningAlgorithm(actions, discount = .9, featureExtractor = features, explorationProb = 0, weights = filename)
    strategy = lambda id,s : rl.getAction(s)
    return strategy