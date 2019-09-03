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
    self.weights = np.array([20,20,20,20,20,20,20,20, 0])
  def act(self, state, reward):
    a0 = np.dot(self.weights, np.append(state,0))
    a1 = np.dot(self.weights, np.append(state,1))
    # Action is weighted random choice  
    a0prob = a0 / (a0 + a1)
    a1prob = a1 / (a0 + a1)
    action = np.random.choice([0,1], p=[a0prob, a1prob])
    return action
  def update_weights(self):
    pass
    
# Initialise the environment and the agent
env = gym.make('CartPole-v0')
agent = BetterAgent(env.action_space)

# Run 10 episodes
iht = tiles.IHT(1024)
normalise = np.array([20,8,30,10]) # adjust observations to be similar sizes
for i in range(10):
  state = env.reset()
  reward = 1
  done = False
  while True:
    env.render()
    state = tiles.tiles(iht, 8, state*normalise)
    action = agent.act(state, reward)
    state, reward, done, _ = env.step(action)
    if done:
      break

# End the session
env.close()