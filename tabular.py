import gym
import numpy as np
# Initialise the Q probabilities
Q = dict()

x = np.array([])
y = np.array([])
runs = 10000

# Start the environment
env = gym.make('CartPole-v0')
action_space = np.array([0,1])
# Run some episodes
for episode in range(runs):
  obs = env.reset()
  a,b,c,d = round(obs[0],0), round(obs[1],1), round(obs[2],1), round(obs[3],1)
  # Greedy action taking
  if (a,b,c,d,0) in Q:
    a0 = Q[(a,b,c,d,0)]
  else:
    Q[(a,b,c,d,0)] = 20
    a0 = 20
  if (a,b,c,d,1) in Q:
    a1 = Q[(a,b,c,d,1)]
  else:
    Q[(a,b,c,d,1)] = 20
    a1 = 20
  # Action is weighted random choice
  a0prob = np.exp(a0) / (np.exp(a0) + np.exp(a1))
  a1prob = np.exp(a1) / (np.exp(a0) + np.exp(a1))
  action = np.random.choice(action_space, p=[a0prob, a1prob])  
  for t in range(200):
    """ Runs one episode until done. """  
    obs, reward, done, info = env.step(action) 
    # Record the observation in the table
    w,x,y,z = round(obs[0],0), round(obs[1],1), round(obs[2],1), round(obs[3],1)
        
    if done:
      Q[(w,x,y,z,action)] = 0
      break # End episode
    
    # Greedy action taking
    if (w,x,y,z,0) in Q:
      a0 = Q[(w,x,y,z,0)]
    else:
      a0 = 20
    if (w,x,y,z,1) in Q:
      a1 = Q[(w,x,y,z,1)]
    else:
      a1 = 20
    # Action is weighted random choice
    if np.random.rand() < 0.9:
      if a0 > a1:
        action_2 = 0
      else:
        action_2 = 1
    else:
      if a1 > a0:
        action_2 = 0
      else:
        action_2 = 1
    
    if (w,x,y,z,action_2) in Q:
      pass
    else:
      Q[(w,x,y,z,action_2)] = 20
    Q[(a,b,c,d,action)] += 0.5 * (1 + Q[(w,x,y,z,action_2)] - Q[(a,b,c,d,action)])
    a,b,c,d = w,x,y,z
    action = action_2

for episode in range(5):
  obs = env.reset()
  a,b,c,d = round(obs[0],0), round(obs[1],1), round(obs[2],1), round(obs[3],1)
  # Greedy action taking
  if (a,b,c,d,0) in Q:
    a0 = Q[(a,b,c,d,0)]
  else:
    Q[(a,b,c,d,0)] = 20
    a0 = 20
  if (a,b,c,d,1) in Q:
    a1 = Q[(a,b,c,d,1)]
  else:
    Q[(a,b,c,d,1)] = 20
    a1 = 20
  # Action is weighted random choice
  a0prob = np.exp(a0) / (np.exp(a0) + np.exp(a1))
  a1prob = np.exp(a1) / (np.exp(a0) + np.exp(a1))
  action = np.random.choice(action_space, p=[a0prob, a1prob])  
  for t in range(200):
    """ Runs one episode until done. """ 
    env.render() 
    obs, reward, done, info = env.step(action) 
    # Record the observation in the table
    w,x,y,z = round(obs[0],0), round(obs[1],1), round(obs[2],1), round(obs[3],1)
        
    if done:
      Q[(w,x,y,z,action)] = 0
      break # End episode
    
    # Greedy action taking
    if (w,x,y,z,0) in Q:
      a0 = Q[(w,x,y,z,0)]
    else:
      a0 = 20
    if (w,x,y,z,1) in Q:
      a1 = Q[(w,x,y,z,1)]
    else:
      a1 = 20
    # Action is weighted random choice
    if np.random.rand() < 0.9:
      if a0 > a1:
        action_2 = 0
      else:
        action_2 = 1
    else:
      if a1 > a0:
        action_2 = 0
      else:
        action_2 = 1
    
    if (w,x,y,z,action_2) in Q:
      pass
    else:
      Q[(w,x,y,z,action_2)] = 20
    Q[(a,b,c,d,action)] += 0.5 * (1 + Q[(w,x,y,z,action_2)] - Q[(a,b,c,d,action)])
    a,b,c,d = w,x,y,z
    action = action_2

  
env.close()
