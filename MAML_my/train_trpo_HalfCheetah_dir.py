import argparse
import os
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from utils import *
from trpo import one_step_trpo,conjugate_gradients,trpo_step

import pickle
import csv

from copy import deepcopy

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
parser.add_argument('--meta-lr', type=float, default=0.1, metavar='G',
                    help='meta lr (default: 0.1)') 
parser.add_argument('--max-kl', type=float, default=3e-3, metavar='G',
                    help='max kl value (default: 3e-2)')
parser.add_argument('--damping', type=float, default=0e-5, metavar='G',
                    help='damping (default: 0e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch-size (default: 20)') 
parser.add_argument('--task-batch-size', type=int, default=4, metavar='N',
                    help='task-batch-size (default: 4)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--max-length', type=int, default=200, metavar='N',
                    help='max length of a path (default: 200)')
args = parser.parse_args()

torch.manual_seed(args.seed)
#if args.env_name=="HalfCheetah-v4":
#    env = gym.make(args.env_name,exclude_current_positions_from_observation=False)
#else:
#    env = gym.make(args.env_name)
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]


running_state = ZFilter((num_inputs,), clip=5)
if os.path.exists("./check_point/running_state_HalfCheetah_dir_maml.pkl"):
    with open("./check_point/running_state_HalfCheetah_dir_maml.pkl",'rb') as file:
        running_state  = pickle.loads(file.read())

print("running_state: ",running_state.rs.n) 


def select_action(state,policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def select_action_test(state,policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    return action_mean

def sample_data_for_task_specific(target_v,policy_net,batch_size):
    memory = Memory()
    memory_extra=Memory()

    accumulated_raward_batch = 0
    num_episodes = 0
    for i in range(batch_size):
        state = env.reset()[0]
        state = running_state(state)

        reward_sum = 0
        for t in range(args.max_length):
            action = select_action(state,policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            reward=info['x_velocity']*target_v-0.5 * 1e-1 * np.sum(np.square(action))
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
            action = select_action(state,policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info= env.step(action)
            reward=info['x_velocity']*target_v-0.5 * 1e-1 * np.sum(np.square(action))
            next_state = running_state(next_state)
            path_number = i

            memory_extra.push(state, np.array([action]), path_number, next_state, reward)
            if args.render:
                env.render()
            state = next_state
            if done or truncated:
                break

        num_episodes += 1
        accumulated_raward_batch += reward_sum

    accumulated_raward_batch /= num_episodes
    batch = memory.sample()
    batch_extra = memory_extra.sample()

    return batch,batch_extra,accumulated_raward_batch


def compute_adavatage(batch,batch_extra,batch_size):
    rewards = torch.Tensor(np.array(batch.reward))
    path_numbers = torch.Tensor(np.array(batch.path_number))
    actions = torch.Tensor(np.array(np.concatenate(batch.action, 0)))
    states = torch.Tensor(np.array(batch.state))

    rewards_extra = torch.Tensor(np.array(batch_extra.reward))
    path_numbers_extra = torch.Tensor(np.array(batch_extra.path_number))
    actions_extra = torch.Tensor(np.array(np.concatenate(batch_extra.action, 0)))
    states_extra = torch.Tensor(np.array(batch_extra.state))

    returns = torch.Tensor(actions.size(0),1)
    prev_return=torch.zeros(batch_size,1)

    k=batch_size-1
    for i in reversed(range(rewards_extra.size(0))):
        if not int(path_numbers_extra[i].item())==k:
            k=k-1
            assert k==path_numbers_extra[i].item()
        prev_return[k,0]=rewards[i]+ args.gamma * prev_return[k,0] 
        
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return[int(path_numbers[i].item()),0]
        prev_return[int(path_numbers[i].item()),0] = returns[i, 0]

    targets = Variable(returns)
    return targets

def task_specific_adaptation_grad(meta_policy_net,batch,q_values,meta_lr): 
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))

    action_means, action_log_stds, action_stds = meta_policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).detach().clone().data

    action_means1, action_log_stds1, action_stds1 = meta_policy_net(Variable(states))
    log_prob = normal_log_density(Variable(actions), action_means1, action_log_stds1, action_stds1)
    aaaa=torch.exp(log_prob - Variable(fixed_log_prob))
    action_loss = -Variable(q_values) *  torch.special.expit(2.0*aaaa-2.0)*2 
    #action_loss = -Variable(q_values) * aaaa
    loss= action_loss.mean()     


    grads_list = torch.autograd.grad(loss, meta_policy_net.parameters(), create_graph=True, retain_graph=True)
    new_parameter_withgrad=[]
    for i,param in enumerate(meta_policy_net.parameters()):
        new_parameter_withgrad.append(list(meta_policy_net.parameters())[i] - meta_lr * grads_list[i])
        new_parameter_withgrad[i].retain_grad()
        
    return new_parameter_withgrad

def task_specific_adaptation_nograd(task_specific_policy,meta_policy_net,batch,q_values,meta_lr): 
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))

    action_means, action_log_stds, action_stds = meta_policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).detach().clone().data

    action_means1, action_log_stds1, action_stds1 = meta_policy_net(Variable(states))
    log_prob = normal_log_density(Variable(actions), action_means1, action_log_stds1, action_stds1)
    aaaa=torch.exp(log_prob - Variable(fixed_log_prob))
    action_loss = -Variable(q_values) *  torch.special.expit(2.0*aaaa-2.0)*2 
    #action_loss = -Variable(q_values) * aaaa
    loss= action_loss.mean()  

    grads_list = torch.autograd.grad(loss, meta_policy_net.parameters(), retain_graph=False)
    for i,param in enumerate(task_specific_policy.parameters()):
        param.data = list(meta_policy_net.parameters())[i].data - meta_lr * grads_list[i].data

    return task_specific_policy

def policy_gradient_obain(task_specific_policy,after_batch,after_q_values):
    actions = torch.Tensor(np.array(np.concatenate(after_batch.action, 0)))
    states = torch.Tensor(np.array(after_batch.state))
    fixed_action_means, fixed_action_log_stds, fixed_action_stds = task_specific_policy(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), fixed_action_means, fixed_action_log_stds, fixed_action_stds).detach().clone().data
    afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds = task_specific_policy(Variable(states))
    log_prob = normal_log_density(Variable(actions), afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds)
    AAAAA=torch.exp(log_prob - Variable(fixed_log_prob))
    #bbbbb=torch.min(Variable(after_q_values)*AAAAA,Variable(after_q_values)*AAAAA*torch.clamp(AAAAA,0.8,1.2))
    bbbbb=Variable(after_q_values)*AAAAA
    #bbbbb=Variable(after_q_values)*torch.special.expit(2.0*AAAAA-2.0)*2
    
    J_loss = (-bbbbb).mean()
    meta_grads_list = torch.autograd.grad(J_loss, task_specific_policy.parameters(), retain_graph=False)
    meta_policy_grad = [grad2.data.clone() for grad2 in meta_grads_list]

    return J_loss, meta_policy_grad


if __name__ == "__main__":
    if not os.path.exists("./check_point/meta_policy_net_HalfCheetah_dir_maml.pkl"):
        meta_policy_net = Policy(num_inputs, num_actions)
    else:
        meta_policy_net = torch.load("./check_point/meta_policy_net_HalfCheetah_dir_maml.pkl")

    "--------------------------------------------------for initialization of running_state------------------------------------------"
    for i in range(args.batch_size*5):
        state = env.reset()[0]
        state = running_state(state)
        for t in range(args.max_length):
            action = select_action(state,meta_policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            next_state = running_state(next_state)

    aaaaaa=-10000

    for i_episode in range(500):
        print("i_episode: ",i_episode)
        meta_lr=args.meta_lr
        print("meta_lr: ",meta_lr)

        batch_list=[]
        q_values1_list=[]
        gradient_main_list=[]
        task_list=[-1.0,1.0,-1.0,1.0]
        
        for task_number in range(args.task_batch_size):
            target_v=task_list[task_number]
            batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,meta_policy_net,args.batch_size)
            print("task_number: ",task_number, " target_v: ", target_v)
            print('(before adaptation) Episode {}\tAverage reward {:.2f}'.format(i_episode, accumulated_raward_batch))
            
            q_values = compute_adavatage(batch,batch_extra,args.batch_size)
            q_values2 = q_values
            q_values1 = (q_values - q_values.mean()) 

            task_specific_policy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation_nograd(task_specific_policy,meta_policy_net,batch,q_values1,meta_lr)

            after_batch,after_batch_extra,after_accumulated_raward_batch=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size*5) 
            print('(after adaptation) Episode {}\tAverage reward {:.2f}'.format(i_episode, after_accumulated_raward_batch)) 

            q_values_after = compute_adavatage(after_batch,after_batch_extra,args.batch_size*5) 
            q_values_after = (q_values_after - q_values_after.mean()) 
            _,gradient_main= policy_gradient_obain(task_specific_policy,after_batch,q_values_after) 

            batch_list.append(batch)
            gradient_main_list.append(gradient_main)
            q_values1_list.append(q_values1)

        
        def get_loss(volatile=False):
            overall_loss=0.0

            for task_number in range(args.task_batch_size):
                
                gradient_main=gradient_main_list[task_number]
                batch=batch_list[task_number]
                q_values12=q_values1_list[task_number]
                new_parameter_withgrad1 = task_specific_adaptation_grad(meta_policy_net,batch,q_values12,meta_lr)
                flat_gradient_main = torch.cat([grad.contiguous().view(-1) for grad in gradient_main])
                flat_new_parameter_withgrad1=torch.cat([param.contiguous().view(-1) for param in new_parameter_withgrad1])
                meta_loss = (flat_gradient_main * flat_new_parameter_withgrad1).sum() 
                overall_loss=overall_loss+1.0/args.task_batch_size* meta_loss

            return overall_loss
        
        states = torch.cat([torch.Tensor(np.array(batch_list[0].state)),torch.Tensor(np.array(batch_list[1].state)),torch.Tensor(np.array(batch_list[2].state)),torch.Tensor(np.array(batch_list[3].state))],dim=0)
        mean101, log_std101, std101 = meta_policy_net(Variable(states))
        mean0 = mean101.clone().detach().data.double()
        log_std0 = log_std101.clone().detach().data.double()
        std0 = std101.clone().detach().data.double()
    
        def get_kl():
            mean1, log_std1, std1 = meta_policy_net(Variable(states))  
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        trpo_step(meta_policy_net, get_loss, get_kl, args.max_kl, args.damping)
        
        target_v_list000=[-1.0,1.0]
        len_target_v_list000=len(target_v_list000)
        result_before=np.zeros(len_target_v_list000)
        result_after=np.zeros(len_target_v_list000)
        for task_number_test in range(len_target_v_list000):
            target_v=target_v_list000[task_number_test]
            batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,meta_policy_net,args.batch_size)
            result_before[task_number_test]=accumulated_raward_batch
    
            q_values = compute_adavatage(batch,batch_extra,args.batch_size) 
            q_values = (q_values - q_values.mean())  

            task_specific_policy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation_nograd(task_specific_policy,meta_policy_net,batch,q_values,meta_lr)

            after_batch,after_batch_extra,after_accumulated_raward_batch=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size)
            result_after[task_number_test]=after_accumulated_raward_batch

        print("result_before: ",result_before.mean())
        print("result_after: ",result_after.mean())
        with open('./check_point/training_log_HalfCheetah_dir_maml.csv', 'a+') as file:
            writer = csv.writer(file)
            writer.writerow([i_episode, result_after.mean()])

        if result_after.mean()>aaaaaa:
            print("save model")
            aaaaaa=result_after.mean()
            torch.save(meta_policy_net, "./check_point/meta_policy_net_HalfCheetah_dir_maml.pkl")
            output_hal = open("./check_point/running_state_HalfCheetah_dir_maml.pkl", 'wb')
            str1 = pickle.dumps(running_state)
            output_hal.write(str1)
            output_hal.close()
        

        print(torch.exp(meta_policy_net.action_log_std)) 

