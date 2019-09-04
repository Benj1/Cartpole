import gym
import numpy as np
import tiles

class RandomAgent(object):
  """ The World's Worst Agent. """
  def __init__(self, action_space):
    self.action_space = action_space
  def act(self, observation, reward):
    return self.action_space.sample()

class BetterAgent(object):
  def __init__(self, action_space):
    self.action_space = action_space
    self.no_of_tiles = 20
    self.weights = np.zeros((200, self.no_of_tiles))
    self.iht = tiles.IHT(200)
    self.normalise = np.array([4,1,1,1,10])
    self.training_rate = 0.1

  def act(self, state, reward):
    x0 = tiles.tiles(self.iht, self.no_of_tiles, np.append(state, 0)*self.normalise)
    x1 = tiles.tiles(self.iht, self.no_of_tiles, np.append(state, 1)*self.normalise)
    a0 = sum([self.weights[x0[i]][i] for i in range(self.no_of_tiles)])
    a1 = sum([self.weights[x1[i]][i] for i in range(self.no_of_tiles)])
    if np.random.rand() < 0.95:
      if a0 > a1:
        action = 0
      else:
        action = 1
    else:
      if a1 > a0:
        action = 0
      else:
        action = 1
    return action

  def update_weights(self, reward, state, new_state, action, new_action):
    x = tiles.tiles(self.iht, self.no_of_tiles, np.append(state, action)*self.normalise)
    new_x = tiles.tiles(self.iht, self.no_of_tiles, np.append(new_state, new_action)*self.normalise)
    q = sum([self.weights[x[i]][i] for i in range(self.no_of_tiles)])
    new_q = sum([self.weights[new_x[i]][i] for i in range(self.no_of_tiles)])
    for i in range(self.no_of_tiles):
      self.weights[x[i]][i] += self.training_rate*(reward + new_q - q)
  
  def terminal_update(self, reward, state, action):
    x = tiles.tiles(self.iht, self.no_of_tiles, np.append(state, action)*self.normalise)
    q = sum([self.weights[x[i]][i] for i in range(self.no_of_tiles)])
    for i in range(self.no_of_tiles):
      self.weights[x[i]][i] += self.training_rate*(reward - q)

# Initialise the environment and the agent
env = gym.make('CartPole-v0')
agent = BetterAgent(env.action_space)

# Run episodes
for i in range(2000):
  state = env.reset()
  reward = 1
  done = False
  action = agent.act(state, reward)
  while True:
    if i > 1900:
      env.render()  
    new_state, reward, done, _ = env.step(action)
    if done:
      agent.terminal_update(reward, state, action)
      break
    new_action = agent.act(new_state, reward)
    agent.update_weights(reward, state, new_state, action, new_action)
    action = new_action
    state = new_state

# End the session
env.close()