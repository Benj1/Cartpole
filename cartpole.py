import gym
import numpy

class RandomAgent(object):
  """ The World's Worst Agent. """
  def __init__(self, action_space):
    self.action_space = action_space
  def act(self, observation, reward, done):
    return self.action_space.sample()

class BetterAgent(object):
  def __init__(self, action_space):
    self.action_space = action_space
  def action(self, observation, reward, done):
    pass

# Initialise the environment and the agent
env = gym.make('CartPole-v0')
agent = RandomAgent(env.action_space)

# Run 10 episodes
for i in range(10):
  ob = env.reset()
  reward = 1
  done = False
  while True:
    env.render()
    action = agent.act(ob, reward, done)
    ob, reward, done, _ = env.step(action)
    if done:
      break
    
# End the session
env.close()