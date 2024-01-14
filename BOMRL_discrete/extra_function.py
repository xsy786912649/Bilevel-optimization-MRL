
import collections
import copy
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
nrow=4
ncol=4


def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def flatten(l):
    return [item for sublist in l for item in sublist]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def softmax_policy_model(observation, policy_model):
    """Compute softmax values for each sets of scores in x."""
    x = policy_model[observation,:]
    probs = softmax(x)
    probs = probs/sum(probs)
    return probs

def sample_actions(observation, policy_model):
    x = softmax_policy_model(observation, policy_model)
    probabilities = x.tolist()
    probabilities=np.array(probabilities)
    probabilities /= probabilities.sum()
    action = np.random.choice(np.arange(0, 4), p = probabilities) 
    return action 

def step(action, env):
    new_state, _,_, _, _ = env.step(action)
    r = int(new_state/4)
    c = new_state%4
    desc = env.desc
    
    reward = 0.0
    if desc[r,c] == b'H':
        reward=-1.0
    elif desc[r,c] == b'G':
        reward=1.0
    else:
        reward=0.0

    return new_state, reward

def sample_trajectorie(env, gamma, policy_model, length=49, episodes=500, beta=0.3):
    # Sample trajectories
    paths = []
    episodes_so_far = 0
    qtable_reward=torch.zeros((16, 4))

    while episodes_so_far < episodes:
        
        episodes_so_far += 1
        states, actions, rewards = [], [], []
        state,_ = env.reset()
        length_so_far = 0
        
        while length_so_far < length:
            length_so_far+=1
            states.append(state)
            action = sample_actions(state, policy_model)
            new_state, reward = step(action, env)
            actions.append(action)
            rewards.append(reward)

            # Update Q(s,a) for the reward
            x = softmax_policy_model(new_state, policy_model)
            probabilities = x.tolist()
            v_new_state= qtable_reward[new_state, 0]*probabilities[0]+qtable_reward[new_state, 1]*probabilities[1]+qtable_reward[new_state, 2]*probabilities[2]+qtable_reward[new_state, 3]*probabilities[3]

            qtable_reward[state, action] = qtable_reward[state, action] + \
                                    beta * (reward + gamma * v_new_state - qtable_reward[state, action])
    
            # Update our current state
            state = new_state
          
        states.append(state)
        state,_ = env.reset()

        path = {"observations": states,
                "actions": actions,
                "rewards": rewards}
        paths.append(path)

    observations=torch.zeros((16))
    for path in paths:
        for i,state in enumerate(path["observations"]):
            observations[state]+=pow(gamma,i)*1.0
    observations=observations/torch.sum(observations)
    observations=observations.reshape((-1,1))

    reward_paths=[discount(path["rewards"], gamma) for path in paths]
    total_reward_paths=np.array(reward_paths)[:,0]
    total_reward = sum(total_reward_paths) / episodes    

    #actions = flatten([path["actions"] for path in paths]) 

    return total_reward, observations, qtable_reward

