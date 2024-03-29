import gym
import numpy as np
import tiles
import torch

class Agent(object):
  """ A superclass for all agents """
  def __init__(self, environment):
    self.environment = environment
  def act(self, observation, reward):
    raise NotImplementedError
  def update_weights(self, reward, state, new_state, action, new_action):
    pass
  def terminal_update(self, reward, state, action):
    pass

class RandomAgent(Agent):
  """ The World's Worst Agent. """
  def act(self, observation, reward):
    action_space = self.environment.action_space
    return action_space.sample()

class BetterAgent(object):
  """ Learns using tile coding and linear function approximation. """
  def __init__(self, action_space):
    self.action_space = action_space
    self.no_of_tiles = 5
    self.weights = np.full((3000, self.no_of_tiles), 5, dtype = float)
    self.iht = tiles.IHT(3000)
    self.normalise = np.array([4,1,1,1,10])
    self.training_rate = 0.1

  def act(self, state, reward):
    """ Epsilon-greedy policy. """
    x0 = tiles.tiles(self.iht, self.no_of_tiles, np.append(state, 0)*self.normalise)
    x1 = tiles.tiles(self.iht, self.no_of_tiles, np.append(state, 1)*self.normalise)
    a0 = sum([self.weights[x0[i]][i] for i in range(self.no_of_tiles)])
    a1 = sum([self.weights[x1[i]][i] for i in range(self.no_of_tiles)])
    if np.random.rand() < 0.9:
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
    """ Update the agents weights after a non-terminal step. """
    x = tiles.tiles(self.iht, self.no_of_tiles, np.append(state, action)*self.normalise)
    new_x = tiles.tiles(self.iht, self.no_of_tiles, np.append(new_state, new_action)*self.normalise)
    q = sum([self.weights[x[i]][i] for i in range(self.no_of_tiles)])
    new_q = sum([self.weights[new_x[i]][i] for i in range(self.no_of_tiles)])
    for i in range(self.no_of_tiles):
      self.weights[x[i]][i] += self.training_rate*(reward + new_q - q)
  
  def terminal_update(self, reward, state, action):
    """ Update the weights after a terminal step. """
    x = tiles.tiles(self.iht, self.no_of_tiles, np.append(state, action)*self.normalise)
    q = sum([self.weights[x[i]][i] for i in range(self.no_of_tiles)])
    for i in range(self.no_of_tiles):
      self.weights[x[i]][i] += self.training_rate*(reward - q)

class IntelligentAgent(object):
  """ Learns using an artifical neural network. """
  def __init__(self, environment):
    self.environment = environment
    self.D_in, self.H1, self.H2, self.D_out = 5, 50, 50, 1
    self.model = torch.nn.Sequential(
        torch.nn.Linear(self.D_in, self.H1),
        torch.nn.ReLU(),
        torch.nn.Linear(self.H1, self.H2),
        torch.nn.ReLU(),
        torch.nn.Linear(self.H2, self.D_out)
    )
    self.learning_rate = 0.1

  def act(self, state, reward, episode):
    """ Greedy action based on feed forward of neural network. """
    state0 = torch.from_numpy(np.append(state, 0)).float()
    state1 = torch.from_numpy(np.append(state, 1)).float()
    q0 = self.model(state0).item()
    q1 = self.model(state1).item()
    if np.random.rand() > 1 - 1/(0.1*episode + 0.1):
      action = np.random.choice([0,1])
    else:
      if q0 > q1:
        action = 0
      else:
        action = 1
    return action
  
  def update_weights(self, reward, state, new_state, action, new_action):
    old_in = torch.from_numpy(np.append(state, action)).float()
    new_in = torch.from_numpy(np.append(new_state, new_action)).float()

    q = self.model(old_in)
    update_target = torch.tensor([reward]) + self.model(new_in)
    loss = (update_target - q)**2

    self.model.zero_grad()
    loss.backward()
    with torch.no_grad():
      for param in self.model.parameters():
        param -= self.learning_rate * param.grad

  def terminal_update(self, reward, state, action):
    input_tensor = torch.from_numpy(np.append(state, action)).float()
    q = self.model(input_tensor)
    update_target = torch.tensor([reward])
    loss = (update_target - q)**2
    self.model.zero_grad()    
    loss.backward()
    with torch.no_grad():
      for param in self.model.parameters():
        param -= self.learning_rate * param.grad

# Initialise the environment and the agent
env = gym.make('CartPole-v0')
agent = IntelligentAgent(env)

# Run episodes
for episode in range(2000):
  state = env.reset()
  reward = 1
  action = agent.act(state, reward, episode)
  while True:
    if episode > 600:
      env.render()  
    new_state, reward, done, _ = env.step(action)
    if done:
      agent.terminal_update(reward, state, action)
      break
    new_action = agent.act(new_state, reward, episode)
    agent.update_weights(reward, state, new_state, action, new_action)
    action = new_action
    state = new_state

# End the session
env.close()