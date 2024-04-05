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
parser.add_argument('--env-name', default="Ant-v4", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--meta-reg', type=float, default=0.001, metavar='G',
                    help='meta regularization regression (default: 1.0)') 
parser.add_argument('--meta-lambda', type=float, default=5.0, metavar='G', 
                    help='meta meta-lambda (default: 0.5)')  
parser.add_argument('--max-kl', type=float, default=3e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=0e-5, metavar='G',
                    help='damping (default: 0e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                    help='batch-size (default: 40)') 
parser.add_argument('--task-batch-size', type=int, default=4, metavar='N',
                    help='task-batch-size (default: 4)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--index', type=int, default=1, metavar='N',
                    help='index (default: 1)')
parser.add_argument('--max-length', type=int, default=200, metavar='N',
                    help='max length of a path (default: 200)')
parser.add_argument('--lower-opt', type=str, default="Adam", metavar='N',
                    help='lower-opt (default: Adam)')
args = parser.parse_args()

torch.manual_seed(args.seed)
#if args.env_name=="Ant-v4":
#    env = gym.make(args.env_name,exclude_current_positions_from_observation=False)
#else:
#    env = gym.make(args.env_name)
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

index = args.index

model_lower="Adam"
if args.lower_opt=="Adam":
    model_lower="Adam"
elif args.lower_opt=="adagrad":
    model_lower="Adagrad"
elif args.lower_opt=="rmsprop":
    model_lower="RMSprop"
elif args.lower_opt=="sgd":
    model_lower="SGD"

running_state = ZFilter((num_inputs,), clip=5)
if os.path.exists("./check_point/running_state_Ant_dir_"+str(index)+".pkl"):
    with open("./check_point/running_state_Ant_dir_"+str(index)+".pkl",'rb') as file:
        running_state  = pickle.loads(file.read())

print(model_lower, "running_state: ",running_state.rs.n) 
print("index: ", index)

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
            next_state, reward_ori, done, truncated, info = env.step(action)
            reward=info['x_velocity']*target_v+0.05+info["reward_survive"]+info["reward_ctrl"]* 1e-2
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
            next_state, reward_ori, done, truncated, info = env.step(action)
            reward=info['x_velocity']*target_v+0.05+info["reward_survive"]+info["reward_ctrl"]* 1e-2
            next_state = running_state(next_state)
            path_number = i

            memory_extra.push(state, np.array([action]), path_number, next_state, reward)
            if args.render:
                env.render()
            state = next_state
            if (done or truncated) and t>0:
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
        prev_return[k,0]=rewards_extra[i]+ args.gamma * prev_return[k,0] 
        
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return[int(path_numbers[i].item()),0]
        prev_return[int(path_numbers[i].item()),0] = returns[i, 0]

    targets = Variable(returns)
    return targets

def task_specific_adaptation(task_specific_policy,meta_policy_net_copy,batch,q_values,meta_lambda_now,index): 
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))

    action_means, action_log_stds, action_stds = meta_policy_net_copy(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).detach().clone().data

    def get_loss():
        action_means1, action_log_stds1, action_stds1 = task_specific_policy(Variable(states))
        log_prob = normal_log_density(Variable(actions), action_means1, action_log_stds1, action_stds1)
        aaaa=torch.exp(log_prob - Variable(fixed_log_prob))
        action_loss = -Variable(q_values) *  torch.special.expit(2.0*aaaa-2.0)*2 
        #action_loss = -Variable(q_values) * aaaa
        return action_loss.mean()     

    def get_kl():
        mean1, log_std1, std1 = task_specific_policy(Variable(states))
        mean_previous, log_std_previous, std_previous = meta_policy_net_copy(Variable(states))

        mean0 = mean_previous.clone().detach().data.double()
        log_std0 = log_std_previous.clone().detach().data.double()
        std0 = std_previous.clone().detach().data.double()

        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
    def get_kl2():
        mean1, log_std1, std1 = task_specific_policy(Variable(states))
        mean_previous, log_std_previous, std_previous = meta_policy_net_copy(Variable(states))

        mean0 = mean_previous.clone().detach().data.double()
        log_std0 = log_std_previous.clone().detach().data.double()
        std0 = std_previous.clone().detach().data.double()

        kl = log_std0 - log_std1 + (std1.pow(2) + (mean1 - mean0).pow(2)) / (2.0 * std0.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
    def get_kl3():
        policy_dictance=torch.tensor(0.0)
        for i,param in enumerate(task_specific_policy.parameters()):
            policy_dictance += (param-list(meta_policy_net_copy.parameters())[i].clone().detach().data).pow(2).sum() 
        return policy_dictance
    if index==1:
        one_step_trpo(task_specific_policy, get_loss, get_kl,meta_lambda_now,args.lower_opt) 
    elif index==2:
        one_step_trpo(task_specific_policy, get_loss, get_kl2,meta_lambda_now,args.lower_opt) 
    elif index==3:
        one_step_trpo(task_specific_policy, get_loss, get_kl3,meta_lambda_now,args.lower_opt) 

    return task_specific_policy

def kl_divergence(meta_policy_net1,task_specific_policy1,batch,index):
    if index==1:
        states = torch.Tensor(np.array(batch.state))
        mean1, log_std1, std1 = task_specific_policy1(Variable(states))
        mean0, log_std0, std0 = meta_policy_net1(Variable(states))
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True).mean()
    elif index==2:
        states = torch.Tensor(np.array(batch.state))
        mean1, log_std1, std1 = task_specific_policy1(Variable(states))
        mean0, log_std0, std0 = meta_policy_net1(Variable(states))
        kl = log_std0 - log_std1 + (std1.pow(2) + (mean1 - mean0).pow(2)) / (2.0 * std0.pow(2)) - 0.5
        return kl.sum(1, keepdim=True).mean()
    elif index==3:
        policy_dictance=torch.tensor(0.0)
        for param,param1 in zip(task_specific_policy1.parameters(),meta_policy_net1.parameters()):
            policy_dictance += (param-param1).pow(2).sum() 
        return policy_dictance

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
    for param in task_specific_policy.parameters():
        param.grad.zero_()
    J_loss.backward(retain_graph=False)
    policy_grad = [param2.grad.data.clone() for param2 in task_specific_policy.parameters()]

    return J_loss, policy_grad

def loss_obain_new(task_specific_policy,meta_policy_net_copy,after_batch,after_q_values):
    actions = torch.Tensor(np.array(np.concatenate(after_batch.action, 0)))
    states = torch.Tensor(np.array(after_batch.state))
    fixed_action_means, fixed_action_log_stds, fixed_action_stds = meta_policy_net_copy(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), fixed_action_means, fixed_action_log_stds, fixed_action_stds).detach().data.clone()
    afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds = task_specific_policy(Variable(states))
    log_prob = normal_log_density(Variable(actions), afteradap_action_means, afteradap_action_log_stds, afteradap_action_stds)
    aaaaa=torch.exp(log_prob - Variable(fixed_log_prob))
    J_loss = (-Variable(after_q_values) * torch.special.expit(2.0*aaaaa-2.0)*2 ).mean()
    #J_loss = (-Variable(after_q_values) * aaaaa).mean()
    
    return J_loss



if __name__ == "__main__":
    if not os.path.exists("./check_point/meta_policy_net_Ant_dir_"+str(index)+".pkl"):
        meta_policy_net = Policy(num_inputs, num_actions)
    else:
        meta_policy_net = torch.load("./check_point/meta_policy_net_Ant_dir_"+str(index)+".pkl")

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
        meta_lambda_now=args.meta_lambda
        print("meta_lambda_now: ",meta_lambda_now)

        x_list=[]
        task_specific_policy_list=[]
        batch_list=[]

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
            meta_policy_net_copy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            for i,param in enumerate(meta_policy_net_copy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation(task_specific_policy,meta_policy_net_copy,batch,q_values1,meta_lambda_now,index)

            after_batch,after_batch_extra,after_accumulated_raward_batch=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size*5) 
            print('(after adaptation) Episode {}\tAverage reward {:.2f}'.format(i_episode, after_accumulated_raward_batch)) 

            q_values_after = compute_adavatage(after_batch,after_batch_extra,args.batch_size*5) 
            q_values_after = (q_values_after - q_values_after.mean()) 

            kl_phi_theta=kl_divergence(meta_policy_net,task_specific_policy,batch,index)

            _, policy_gradient_main_term= policy_gradient_obain(task_specific_policy,after_batch,q_values_after)

            loss_for_1term=loss_obain_new(task_specific_policy,meta_policy_net,batch,q_values1)
            
            #(\nabla_\phi^2 kl_phi_theta+loss_for_1term) x= policy_gradient_2term
            def d_theta_2_kl_phi_theta_loss_for_1term(v):
                grads = torch.autograd.grad(kl_phi_theta+loss_for_1term/meta_lambda_now, task_specific_policy.parameters(), create_graph=True,retain_graph=True)
                flat_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
                kl_v = (flat_grad_kl * Variable(v)).sum()
                grads_new = torch.autograd.grad(kl_v, task_specific_policy.parameters(), create_graph=True,retain_graph=True)
                flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads_new]).data.clone()
                return flat_grad_grad_kl
            policy_gradient_main_term_flat=torch.cat([grad.contiguous().view(-1) for grad in policy_gradient_main_term]).data
            x = conjugate_gradients(d_theta_2_kl_phi_theta_loss_for_1term, policy_gradient_main_term_flat, 10)
            x_list.append(x.data)
            task_specific_policy_list.append(task_specific_policy)
            batch_list.append(batch)
        
        def get_loss(volatile=False):
            overall_loss=0.0

            for task_number in range(args.task_batch_size):
                
                task_specific_policy=task_specific_policy_list[task_number]
                batch=batch_list[task_number]
                x=x_list[task_number]

                kl_phi_theta_1=kl_divergence(meta_policy_net,task_specific_policy,batch,index)
                grads_1 = torch.autograd.grad(kl_phi_theta_1, task_specific_policy.parameters(), create_graph=True,retain_graph=True)
                flat_grad_kl_1 = torch.cat([grad.contiguous().view(-1) for grad in grads_1])
                kl_v_1 = -(flat_grad_kl_1 * x).sum() 

                overall_loss=overall_loss+kl_v_1*1.0/args.task_batch_size

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
            meta_policy_net_copy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            for i,param in enumerate(meta_policy_net_copy.parameters()):
                param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation(task_specific_policy,meta_policy_net_copy,batch,q_values,meta_lambda_now,index)

            after_batch,after_batch_extra,after_accumulated_raward_batch=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size)
            result_after[task_number_test]=after_accumulated_raward_batch

        print("result_before: ",result_before.mean())
        print("result_after: ",result_after.mean())
        with open('./check_point/training_log_Ant_dir_'+str(index)+'.csv', 'a+') as file:
            writer = csv.writer(file)
            writer.writerow([i_episode, result_after.mean()])

        if result_after.mean()>aaaaaa:
            print("save model")
            aaaaaa=result_after.mean()
            torch.save(meta_policy_net, "./check_point/meta_policy_net_Ant_dir_"+str(index)+".pkl")
            output_hal = open("./check_point/running_state_Ant_dir_"+str(index)+".pkl", 'wb')
            str1 = pickle.dumps(running_state)
            output_hal.write(str1)
            output_hal.close()
        
        #torch.save(meta_policy_net, "./check_point/meta_policy_net_Ant_dir_"+str(i_episode)+".pkl")
        #output_hal = open("./check_point/running_state_Ant_dir_"+str(i_episode)+".pkl", 'wb')
        #str1 = pickle.dumps(running_state)
        #output_hal.write(str1)
        #output_hal.close()

        print(torch.exp(meta_policy_net.action_log_std)) 

