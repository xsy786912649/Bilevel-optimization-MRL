import argparse
from itertools import count
import os

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step,one_step_trpo
from utils import *
from copy import deepcopy

import pickle

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="HalfCheetah-v4", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--meta-lambda', type=float, default=2.0, metavar='G',
                    help='meta meta-lambda (default: 2.0)') 
parser.add_argument('--max-kl', type=float, default=3e-2, metavar='G',
                    help='max kl value (default: 3e-2)')
parser.add_argument('--damping', type=float, default=0e-1, metavar='G',
                    help='damping (default: 0e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch-size (default: 20)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--max-length', type=int, default=200, metavar='N',
                    help='max length of a path (default: 200)')
args = parser.parse_args()

#if args.env_name=="HalfCheetah-v4":
#    env = gym.make(args.env_name,exclude_current_positions_from_observation=False)
#else:
#    env = gym.make(args.env_name)
env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

#env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)

model_lower="Adam"
if not os.path.exists("meta_policy_net_"+model_lower+".pkl"):
    policy_net = Policy(num_inputs, num_actions)
else:
    print("gg")
    policy_net = torch.load("meta_policy_net_"+model_lower+".pkl")

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch,batch_extra,batch_size):
    rewards = torch.Tensor(np.array(batch.reward))
    path_numbers = torch.Tensor(np.array(batch.path_number))
    actions = torch.Tensor(np.array(np.concatenate(batch.action, 0)))
    states = torch.Tensor(np.array(batch.state))

    rewards_extra = torch.Tensor(np.array(batch_extra.reward))
    path_numbers_extra = torch.Tensor(np.array(batch_extra.path_number))
    actions_extra = torch.Tensor(np.array(np.concatenate(batch_extra.action, 0)))
    states_extra = torch.Tensor(np.array(batch_extra.state))

    def update_advantage_function(): 
        
        returns = torch.Tensor(actions.size(0),1)
        prev_return=torch.zeros(batch_size,1)

        k=batch_size-1
        for i in reversed(range(rewards_extra.size(0))):
            if not int(path_numbers_extra[i].item())==k:
                k=k-1
                assert k==path_numbers_extra[i].item()
            prev_return[k,0]=rewards_extra[i]+ args.gamma * prev_return[k,0] 
            
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + args.gamma * prev_return[int(path_numbers[i].item()),0]
            prev_return[int(path_numbers[i].item()),0] = returns[i, 0]
            
        targets = Variable(returns)
        return targets

    q_values=update_advantage_function()

    print(q_values.std())
    print(q_values.mean())
    q_values = (q_values - q_values.mean())
    
    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(q_values) * torch.special.expit(2.0*torch.exp(log_prob - Variable(fixed_log_prob))-2.0)*2
        #action_loss = -Variable(q_values) * torch.exp(log_prob - Variable(fixed_log_prob))

        return action_loss.mean()

    mean101, log_std101, std101 = policy_net(Variable(states))
    mean0 = mean101.clone().detach().data.double()
    log_std0 = log_std101.clone().detach().data.double()
    std0 = std101.clone().detach().data.double()

    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))  
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    #trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)
    one_step_trpo(policy_net, get_loss, get_kl,args.meta_lambda,lower_opt='Adam') 

    print(torch.exp(policy_net.action_log_std))

    return 

running_state = ZFilter((num_inputs,), clip=5)

"--------------------------------------------------for initialization of running_state------------------------------------------"
for i in range(args.batch_size):
    state = env.reset()[0]
    state = running_state(state)
    for t in range(args.max_length):
        action = select_action(state)
        action = action.data[0].numpy()
        next_state, reward, done, truncated, info = env.step(action)
        next_state = running_state(next_state)

if __name__ == "__main__":

    for i_episode in count(1):
        memory = Memory()
        memory_extra=Memory()

        reward_batch = 0
        num_episodes = 0
        for i in range(args.batch_size):
            state = env.reset()[0]
            state = running_state(state)

            reward_sum = 0
            for t in range(args.max_length):
                action = select_action(state)
                action = action.data[0].numpy()
                next_state, reward, done, truncated, info = env.step(action)
                reward=-abs(info['x_velocity']-1.5)#-0.5 * 1e-1 * np.sum(np.square(action))
                reward_sum += reward
                next_state = running_state(next_state)
                path_number = i

                memory.push(state, np.array([action]), path_number, next_state, reward)
                if args.render:
                    env.render()
                state = next_state
                if done or truncated:
                    break
            
            env._elapsed_steps=0
            for t in range(args.max_length):
                action = select_action(state)
                action = action.data[0].numpy()
                next_state, reward, done, truncated, info = env.step(action)
                reward=-abs(info['x_velocity']-1.5)#-0.5 * 1e-1 * np.sum(np.square(action))
                next_state = running_state(next_state)
                path_number = i

                memory_extra.push(state, np.array([action]), path_number, next_state, reward)
                if args.render:
                    env.render()
                state = next_state
                if done or truncated:
                    break

            num_episodes += 1
            reward_batch += reward_sum

        reward_batch /= num_episodes
        batch = memory.sample()
        batch_extra = memory_extra.sample()
        update_params(batch,batch_extra,args.batch_size)

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))
